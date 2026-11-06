"""Leg 6, S2: the fibre lateral-overlap readout for the cited inputs.

No simulation. The readout is a model evaluation, so this leg runs under `kind: audit` and the
preflight skips the GPU assertion while still enforcing the pin and the snapshot.

Run:  ./tools/run.sh configs/leg6-fibre-overlap.json src/leg6_fibre.py
      ./.venv/bin/python src/leg6_fibre.py <run-dir> --negative-control
"""
from parity import *
import coupling

STAGES = ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout')
STAGE_SOURCE = {
    'F1-structure': 'coupling.CITED_INPUTS (the device width, cited from the paper)',
    'F2-mesh': 'not applicable: no grid is built by this leg',
    'F3-source': 'not applicable: no source is launched by this leg',
    'F4-monitors': 'not applicable: no monitor is read by this leg',
    'F5-run': 'not applicable: no simulation runs in this leg',
    'F6-readout': 'coupling.lateral_overlap (the transcribed overlap model)',
}


def main():
    cfg, run = load_run()
    cited = coupling.CITED_INPUTS
    width = cited['device_width_um']['value']
    waist = cited['fibre_waist_um']['value']
    factor = coupling.lateral_overlap(width, waist)
    table = coupling.sweep(cfg['sweep_widths_um'], cfg['sweep_waists_um'])
    peak = coupling.peak_width(waist)

    measured = {
        'device_width_um': width,
        'fibre_mode_field_diameter_um': cited['fibre_mfd_um']['value'],
        'fibre_waist_um': waist,
        'eta_lat': factor,
        'model_peak_eta_lat': peak['peak_factor'],
        'model_peak_width_um': peak['peak_width_um'],
    }
    readout = dict(
        policy=cfg['policy'], status='completed', simulation_executed=False,
        fixture=('No simulation. This is the lateral overlap of a laterally uniform aperture of '
                 'the cited width with the cited fibre mode. It is one factor of a fibre '
                 'efficiency, not the efficiency.'),
        stages=list(STAGES), stage_source=dict(STAGE_SOURCE),
        cases=[dict(id=cfg['case_id'], ppw=0, steps=0, dt_s=0.0, shape=[0, 0, 0],
                    measured=measured)],
        cited_inputs=cited, sensitivity=table,
        total_efficiency_note=(
            'eta_total = eta_line_focus x eta_lat. eta_line_focus is the 2D line-focus efficiency '
            'of the guided device and is NOT available in this repository, so no total is '
            'reported here. The reference project records 26.16 percent on one fixture and 45.95 '
            'percent on another; those are that project\'s numbers for that project\'s fixtures '
            'and are cited, not used.'),
        model_limitation=(
            'The aperture is laterally uniform, so it spreads power where the mode is not. The '
            'overlap therefore peaks at a finite width and falls for wider devices. A wider device '
            'is not automatically better under this model.'),
    )
    write_json(Path(run) / 'readout.json', readout)

    print(f'cited: W = {width} um, fibre MFD = {cited["fibre_mfd_um"]["value"]} um, '
          f'waist = {waist} um ({cited["fibre_waist_um"]["kind"]})')
    print(f'eta_lat(W={width}, w_lat={waist}) = {factor:.6f}')
    print(f'model peak: {peak["peak_factor"]:.6f} at W = {peak["peak_width_um"]:.3f} um')
    print('sensitivity (never the headline):')
    for row in table:
        label = ' '.join(f'{key}={value:.6f}' for key, value in row.items() if key != 'width_um')
        print(f'  W={row["width_um"]:6g}  {label}')
    print(f'leg6 readout: eta_lat = {factor:.6f}')
    return 0


def negative_control():
    """Three inputs the model must reject rather than return as a plausible number."""
    run = Path(sys.argv[1])
    readout = json.loads((run / 'readout.json').read_text())
    print(f'control 0: the cited inputs give eta_lat = {readout["cases"][0]["measured"]["eta_lat"]:.6f}')
    cases = [('zero-width device', 0.0, 5.2),
             ('negative width', -12.0, 5.2),
             ('zero fibre waist', 12.0, 0.0)]
    rejected = 0
    for name, width, waist in cases:
        try:
            value = coupling.lateral_overlap(width, waist)
            print(f'control FAILED ({name}): returned {value!r} instead of refusing')
        except ValueError as error:
            rejected += 1
            print(f'control ({name}): refused, {error}')
    print('Source run files were not modified.')
    return 0 if rejected == len(cases) else 1


if __name__ == '__main__':
    sys.exit(negative_control() if '--negative-control' in sys.argv else main())
