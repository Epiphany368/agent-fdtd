"""Leg 7, part 1 of S1: the ASM cost rules, re-derived here. Pure arithmetic.

Run:  ./tools/run.sh configs/leg7-asm-feasibility.json src/leg7_asm.py
      ./.venv/bin/python src/leg7_asm.py <run-dir> --negative-control
"""
from parity import *
import oblique_asm

STAGES = ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout')
STAGE_SOURCE = {
    'F1-structure': 'not applicable: no structure is built by this part',
    'F2-mesh': 'not applicable: no grid is built by this part',
    'F3-source': 'oblique_asm.reconstruct (the ASM sum that would build the source)',
    'F4-monitors': 'not applicable',
    'F5-run': 'not applicable: no FDTD runs in this part',
    'F6-readout': 'oblique_asm.reconstruction_error',
}


def main():
    cfg, run = load_run()
    k0 = 2.0 * math.pi / cfg['wavelength_um']
    rows, chosen = [], []
    for waist in cfg['waists_um']:
        first = oblique_asm.first_m_within(waist, cfg['period_um'], k0, cfg['tolerance'],
                                           tuple(cfg['orders']))
        rules = oblique_asm.cost_rules(waist, cfg['period_um'])
        affordable = first is not None and first <= cfg['budget_runs']
        chosen.append(dict(w0_um=waist, first_m_within_tolerance=first, affordable=affordable,
                           **rules))
        rows.extend(oblique_asm.sweep(waist, cfg['period_um'], k0, tuple(cfg['orders'])))
    readout = dict(
        policy=cfg['policy'], status='completed', simulation_executed=False,
        fixture=('Pure arithmetic. The Array Scanning Method reconstruction of a Gaussian beam '
                 'from its plane-wave spectrum over the Brillouin zone. No FDTD.'),
        stages=list(STAGES), stage_source=dict(STAGE_SOURCE),
        cases=[dict(id='asm-feasibility', ppw=0, steps=0, dt_s=0.0, shape=[0, 0, 0],
                    measured=dict(tolerance=cfg['tolerance'], budget_runs=cfg['budget_runs'],
                                  wavelength_um=cfg['wavelength_um'], period_um=cfg['period_um']))],
        cost_rules=chosen, sweep=rows,
        note=('Part 1 of leg 7. Part 2 launches the M Bloch cells this part sizes. This part does '
              'not run FDTD and does not claim any field result.'))
    write_json(Path(run) / 'readout.json', readout)

    print(f'period d = {cfg["period_um"]} um, wavelength = {cfg["wavelength_um"]} um, '
          f'tolerance = {cfg["tolerance"]:g}')
    for entry in chosen:
        print(f'  w0 = {entry["w0_um"]:4.1f} um: beam covers '
              f'{entry["periods_covered_by_beam"]:5.2f} periods; '
              f'window predicts M > {entry["window_prediction"]:5.2f}, '
              f'spectrum predicts M > {entry["spectrum_prediction"]:5.2f}; '
              f'tighter = {entry["tighter_rule"]}; first M within tolerance = '
              f'{entry["first_m_within_tolerance"]}; affordable = {entry["affordable"]}')
    for waist in cfg['waists_um']:
        print(f'  w0 = {waist} um sweep:')
        for row in rows:
            if row['w0_um'] != waist:
                continue
            print(f'    M = {row["m"]:3d}  rel L2 error = {row["rel_l2_error"]:.3e}')
    print(f'leg7 part 1: {len(chosen)} waists sized')
    return 0


def negative_control():
    """Criterion C5: an M too small must fail, and a wrong Brillouin-zone span must fail."""
    run = Path(sys.argv[1])
    readout = json.loads((run / 'readout.json').read_text())
    config = json.loads((run / 'config.input.json').read_text())
    x = np.linspace(-2.0, 2.0, 801)
    k0 = 2.0 * math.pi / config['wavelength_um']
    tolerance = config['tolerance']
    print('control 0 (reference): M = 8, w0 = 2.0 -> '
          f'{oblique_asm.reconstruction_error(x, 2.0, 0.9, 8, k0):.3e}')
    small = oblique_asm.reconstruction_error(x, 2.0, 0.9, 4, k0)
    print(f'control 1 (M = 4, too small): error = {small:.3e}; '
          f'exceeds tolerance = {small > tolerance}')
    wrong = oblique_asm.reconstruction_error(x, 2.0, 0.9, 32, k0, correct_span=False)
    print(f'control 2 (Brillouin zone span halved): error = {wrong:.3e}; '
          f'exceeds tolerance = {wrong > tolerance}')
    print('Source run files were not modified.')
    return 0 if (small > tolerance and wrong > tolerance) else 1


if __name__ == '__main__':
    sys.exit(negative_control() if '--negative-control' in sys.argv else main())
