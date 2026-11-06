# Leg 7 part 2 — the ASM campaign — ATTEMPTED AND FAILED — 2026-10-18

**Verdict: the reconstruction gate and the energy gate both fail. Part 2 is not delivered.** This
file records the failure, its size and the three attempts, so the next attempt starts from evidence
instead of from scratch.

## What ran

`./tools/run.sh configs/leg7-asm-campaign.json src/leg7_campaign.py`, run
`20261018T065827Z-299828`, exit `55`. Eight Bloch cells, one per midpoint of the Brillouin zone, on
the homogeneous cell the feasibility study used. About `4` seconds each, `32` seconds total.

| gate | measured | tolerance | result |
|---|---|---|---|
| reconstruction error against the arithmetic reference | `0.1783` | `0.05` | **FAIL**, 3.6x over |
| delivered power across the swept angles, relative spread | `0.391` | `0.002` | **FAIL**, 196x over |
| fixed-wavevector drift over `1.50`–`1.60` µm | `0.4002` deg | (reported, not gated) | measured |

The tolerance was **not** widened. The configuration was **not** changed to make a gate pass.

## A real measurement that came out of it

The soft source does **not** inject the same power at every angle. Measured, in solver reduced units:

| `k_parallel` (µm⁻¹) | 0 | 0.436 | 1.309 | 2.182 | 3.054 |
|---|---|---|---|---|---|
| delivered power | `9.198e-41` | `9.198e-41` | `8.756e-41` | `7.797e-41` | `6.085e-41` |

A `39 percent` fall from normal to the zone edge. The forward and backward halves are equal to five
figures at every angle, so the split is even; it is the **total** that falls.

## The three attempts, and what each ruled out

| attempt | change | reconstruction error | conclusion |
|---|---|---|---|
| 1 | first version | `1.9211` | the phase grid was built from each swept `M` instead of one registered `M`, and the sum had no cell index at all, so no beam was formed. A driver defect, not physics. |
| 2 | one fixed `M = 8` grid of midpoints; the per-cell phase `exp(i k_m n d)` restored; the energy gate made relative | `0.1819` | the beam now forms. Both gates still fail. |
| 3 | each cell's response divided by the square root of the power that angle delivered | `0.1783` | the angle response is **not** the cause: the change moved the error by `2 percent`. |

## The open question, stated so the next attempt does not re-derive it

**The soft plane source spans one cell, `0.9` µm wide.** The weights `A(k) = exp(-k^2 w0^2 / 4)`
shape the *angular spectrum* of the sum, but the aperture that is actually excited is the cell, not a
`4` µm-wide beam. The reconstruction is therefore compared against a `w0 = 2` µm Gaussian while the
aperture that produced it is `0.9` µm across, and the monitor sits `1.1` µm from the source, in the
near field where the periodic array's replicas have not decohered.

So the next attempt must establish, **before** running, which of these the method actually needs:

1. an aperture that spans the beam (a source of width `2 w0`, not one cell), or
2. a reconstruction compared against the diffraction of a one-cell aperture rather than a Gaussian
   of waist `w0`, or
3. a monitor far enough away that the replicas have decohered.

That is a question about the method, not about the solver, and it is cheaper to settle by arithmetic
than by another campaign.

## The frequency drift, measured

A fixed Bloch wavevector holds `k_parallel` constant, so the propagation angle falls as the
wavelength rises. Over `1.50` to `1.60` µm the drift is `0.4002` degrees at the middle wavevector
used here. **The ASM sum uses `k`-independent weights, so it does not remove this drift** — it makes
the beam finite, nothing more. The drift is the route's error bound across the band, and it must be
either compensated by a frequency-dependent weight or accepted and declared. This was criterion C2
and it is now measured.

---

## Second round of attempts, after the method arithmetic — all four fail

The owner asked for a re-run. Four attempts have now been made in total. **None passes, and the
re-runs eliminated the hypotheses without producing a working reconstruction.**

| attempt | change | reconstruction error | verdict |
|---|---|---|---|
| 4 | reconstructions re-derived with the centred cell window; compared `x = i d`, `x = i d - d/2`, `x = i d + d/2` against the sampled Gaussian | `0.0894` / `0.255` / `0.254`, all **flat** in `M` | arithmetic, no run. The driver's **midpoint phase grid** makes the weights anti-periodic, so no simple shift recovers the Gaussian. The `fft.ifft` form converges; the midpoint form does not. |
| 5 | the intra-cell Bloch phase `exp(i k xi)` removed from each monitor profile before summing, so the sum carries only the cell-index phase `exp(i k n d)` | `0.2617` | worse than attempt 2, so this hypothesis is not the fix either |

The energy gate never moved: `0.391` relative, in every attempt.

## Stop. The assumption that is probably wrong.

Four failures with different hypotheses each time means the wrong assumption is upstream of all of
them. The assumption is:

> that this configuration — a homogeneous Bloch cell, a soft plane source one cell wide, a monitor
> `1.1` µm away, `6000` steps — is a valid place to test the ASM sum at all.

Nothing established that. The method is normally applied where the observable spans **more than one
period**: either a region wider than the cell, or a far field where the array's replicas appear as
separated grating orders and the `k`-sum can extinguish all but one. A one-period monitor inside the
source cell sees the superposition of every replica, and the sum may simply not be able to unmix it.

The delivered-power spread of `39 percent` between normal incidence and the zone edge is a second
sign that the linear-combination premise does not hold cleanly in this configuration: the response
*shape* may be changing with `k`, not just its amplitude.

## The cheapest decisive test, stated but not run

Test the `k`-sum on a scalar problem with an exact answer and no FDTD: a one-dimensional periodic
medium with a single-cell source at `k = 0`, where the array response is known in closed form. If
the sum cannot reconstruct a single source there, the method needs a different observable, not a
different driver. If it can, then the FDTD configuration is the problem, and specifically the soft
source's `k`-dependent response.

**This has not been run.** It is the recommended next move, and it costs minutes.
