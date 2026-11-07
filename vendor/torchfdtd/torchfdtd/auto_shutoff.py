"""Native energy + spectral auto-shutoff with matched discrete-adjoint horizon.

Monitors domain electromagnetic energy decay and online DFT spectral convergence
during ``_FDTD.forward``. Once both the energy ratio (relative to peak pulse
energy) and relative spectral change between consecutive check intervals remain
below their tolerances for ``consecutive`` checks after all active sources have
ended and ``min_time_fs`` has elapsed, time stepping terminates early.

``system.region`` and ``spectral.steps`` are updated to ``actual_steps`` so that
any subsequent backward pass replays the exact realized time horizon.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np
import torch


def _validate_shutoff_params(
    min_time_fs: float,
    check_interval: int,
    energy_tolerance: float,
    spectral_tolerance: float,
    consecutive: int,
) -> tuple[float, int, float, float, int]:
    if isinstance(min_time_fs, bool) or not isinstance(min_time_fs, (int, float)) or not math.isfinite(min_time_fs) or min_time_fs < 0:
        raise ValueError('min_time_fs must be a finite non-negative number.')
    if isinstance(check_interval, bool) or not isinstance(check_interval, int) or check_interval < 1:
        raise ValueError('check_interval must be a positive integer.')
    if isinstance(energy_tolerance, bool) or not isinstance(energy_tolerance, (int, float)) or not math.isfinite(energy_tolerance) or energy_tolerance <= 0:
        raise ValueError('energy_tolerance must be a positive finite number.')
    if isinstance(spectral_tolerance, bool) or not isinstance(spectral_tolerance, (int, float)) or not math.isfinite(spectral_tolerance) or spectral_tolerance <= 0:
        raise ValueError('spectral_tolerance must be a positive finite number.')
    if isinstance(consecutive, bool) or not isinstance(consecutive, int) or consecutive < 1:
        raise ValueError('consecutive must be a positive integer.')
    return (
        float(min_time_fs),
        int(check_interval),
        float(energy_tolerance),
        float(spectral_tolerance),
        int(consecutive),
    )


def install_auto_shutoff(
    *,
    min_time_fs: float = 1200.0,
    check_interval: int | None = None,
    interval: int = 2048,
    energy_tolerance: float = 1e-6,
    spectral_tolerance: float = 2e-5,
    consecutive: int = 3,
    callback: Callable[[dict[str, Any]], None] | None = None,
):
    """Install energy + spectral auto-shutoff on ``_FDTD.forward`` and return original."""
    eff_interval = interval if check_interval is None else check_interval
    min_time_fs, eff_interval, energy_tolerance, spectral_tolerance, consecutive = (
        _validate_shutoff_params(
            min_time_fs, eff_interval, energy_tolerance, spectral_tolerance, consecutive
        )
    )
    from .differentiable import _FDTD

    original = _FDTD.forward

    def forward(ctx, epsilon, system, options, report, spectral):
        if spectral is None or spectral.window is not None or system.pmc:
            raise ValueError(
                'Auto-shutoff requires spectral, unapodized, non-PMC observations.'
            )
        if system.region.complex_fields or hasattr(system, 'poles'):
            raise ValueError(
                'Auto-shutoff is validated only for real nondispersive Yee fields.'
            )
        ctx.save_for_backward(epsilon)
        ctx.system, ctx.options, ctx.report = system, options, report
        ctx.spectral = spectral
        maximum = system.region.steps
        dt = system.region.time_step
        block = spectral.block_size
        step_interval = math.ceil(eff_interval / block) * block
        min_steps = math.ceil(min_time_fs * 1e-15 / dt)

        # A nonzero source must have ended; continuous sources cannot terminate early.
        active_source_end = 0
        for family in ('E', 'H'):
            for _, _, wave, _ in system.sources[family]:
                peak = wave.abs().max()
                if float(peak) > 0:
                    nonzero = torch.nonzero(wave.abs() > peak * 1e-8)
                    if len(nonzero):
                        active_source_end = max(active_source_end, int(nonzero[-1]) + 1)

        widths = [
            torch.as_tensor(np.diff(x).copy(), dtype=epsilon.dtype, device=epsilon.device)
            for x in system.region.mesh_nodes
        ]
        volume = widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :]
        # Normalize geometric cell volume factor; only energy ratios are used.
        volume = volume / volume.max()
        material = system._volume(epsilon.detach())

        def energy() -> float:
            value = epsilon.new_zeros(())
            for component in range(3):
                eps = material if material.ndim == 3 else material[..., component]
                value += (
                    (
                        eps * system.grid.E[..., component].square()
                        + system.grid.H[..., component].square()
                    )
                    * volume
                ).sum()
            return float(value)

        system.zero()
        signals = spectral.zeros()
        samples = system.grid.E.new_empty((block, len(system.monitors)))
        started = time.perf_counter()
        peak_energy = 0.0
        previous = None
        stable = 0
        checks: list[dict[str, Any]] = []
        # Sample more frequently during the active source pulse so peak energy is captured.
        source_check = max(block, math.ceil(max(active_source_end, 1) / 8 / block) * block)
        actual = maximum
        converged = False
        current_energy = 0.0

        for step in range(maximum):
            system.advance(step, step + 1)
            local = step % block
            samples[local] = system.observe(system.state())
            if local + 1 == block or step + 1 == maximum:
                spectral.accumulate(signals, samples[: local + 1], step - local)
            count = step + 1
            check = count % step_interval == 0 or count == maximum
            if check or (
                count <= active_source_end + source_check and count % source_check == 0
            ):
                current_energy = energy()
                if not math.isfinite(current_energy):
                    raise RuntimeError('Nonfinite FDTD field energy')
                peak_energy = max(peak_energy, current_energy)
            if check:
                # Divide out SI seconds before norm/squaring to prevent FP32 underflow.
                snapshot = (signals / dt).detach().clone()
                spectral_change = None
                if previous is not None:
                    scale = float(torch.linalg.vector_norm(snapshot))
                    spectral_change = float(
                        torch.linalg.vector_norm(snapshot - previous)
                    ) / max(scale, 1e-30)
                ratio = current_energy / max(peak_energy, 1e-30)
                stable = (
                    stable + 1
                    if (
                        count >= max(min_steps, active_source_end)
                        and ratio <= energy_tolerance
                        and spectral_change is not None
                        and spectral_change <= spectral_tolerance
                    )
                    else 0
                )
                item = dict(
                    steps=count,
                    time_fs=count * dt * 1e15,
                    energy_ratio=ratio,
                    spectral_relative_change=spectral_change,
                    stable_checks=stable,
                    elapsed_s=time.perf_counter() - started,
                )
                checks.append(item)
                if callback is not None:
                    callback(item)
                previous = snapshot
                if stable >= consecutive:
                    actual = count
                    converged = True
                    break

        if epsilon.is_cuda:
            torch.cuda.synchronize(epsilon.device)
        # Backward replays this exact realized horizon.
        system.region = system.region.model_copy(update={'steps': actual})
        spectral.steps = actual
        report.update(
            forward_seconds=time.perf_counter() - started,
            maximum_steps=maximum,
            actual_steps=actual,
            tail_converged=converged,
            early_stop_checks=checks,
            tail_energy_tolerance=energy_tolerance,
            tail_spectral_tolerance=spectral_tolerance,
            realized_horizon_gradient='fixed cutoff, matched discrete transpose',
        )
        return signals

    _FDTD.forward = staticmethod(forward)
    return original


def restore_auto_shutoff(original) -> None:
    """Restore the original ``_FDTD.forward`` staticmethod."""
    from .differentiable import _FDTD

    _FDTD.forward = staticmethod(original)


# Aliases matching research/cloud_fast_stop.py
install = install_auto_shutoff
restore = restore_auto_shutoff


class AutoShutoff:
    """Context manager and controller for energy + spectral FDTD auto-shutoff."""

    def __init__(
        self,
        *,
        min_time_fs: float = 1200.0,
        check_interval: int | None = None,
        interval: int = 2048,
        energy_tolerance: float = 1e-6,
        spectral_tolerance: float = 2e-5,
        consecutive: int = 3,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        eff_interval = interval if check_interval is None else check_interval
        (
            self.min_time_fs,
            self.check_interval,
            self.energy_tolerance,
            self.spectral_tolerance,
            self.consecutive,
        ) = _validate_shutoff_params(
            min_time_fs, eff_interval, energy_tolerance, spectral_tolerance, consecutive
        )
        self.callback = callback
        self.checks: list[dict[str, Any]] = []
        self._original = None

    def _record(self, item: dict[str, Any]) -> None:
        self.checks.append(item)
        if self.callback is not None:
            self.callback(item)

    def __enter__(self) -> AutoShutoff:
        self.checks.clear()
        self._original = install_auto_shutoff(
            min_time_fs=self.min_time_fs,
            check_interval=self.check_interval,
            energy_tolerance=self.energy_tolerance,
            spectral_tolerance=self.spectral_tolerance,
            consecutive=self.consecutive,
            callback=self._record,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._original is not None:
            restore_auto_shutoff(self._original)
            self._original = None


__all__ = [
    'AutoShutoff',
    'install_auto_shutoff',
    'restore_auto_shutoff',
    'install',
    'restore',
]
