"""G7-01r4: constrained three-ridge design, held-out seeds 7, 8 and 9.

Six Adam steps at 0.02 um/560 fs select the better initial or final hard design.
Evaluation uses 0.01/0.005 um, 2240 fs, 41 wavelengths and TE/TM normal/Bloch
conditions, followed by a converged TORCWA comparison. The initializer used
TORCWA-assisted development. Checkpoint replay is not supported.
"""
import argparse

from . import workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--stage', choices=('all', 'seed', 'baseline', 'rcwa', 'judge'), default='all')
    parser.add_argument('--seed', type=int, choices=(7, 8, 9))
    parser.add_argument('--rcwa-python', default=None)
    parser.add_argument('--skip-rcwa', action='store_true',
                        help='run FDTD first; stage rcwa must complete before the final judgement')
    args = parser.parse_args(argv)
    if args.stage == 'seed' and args.seed is None:
        parser.error('--stage seed needs --seed')
    path = workflow.ROOT/'docs/validation/cases/G7-01r4.json'
    case, _ = workflow.load_json(path)
    revision = case.get('revision', {})
    if (case.get('declared_before_run') is not True
            or not revision.get('approved_by')
            or 'draft_status' in case):
        raise RuntimeError(
            'G7-01r4 is a draft. Owner approval and a committed declaration '
            'are required before any declared workflow stage runs.')
    forwarded = ['--out', args.out, '--stage', args.stage]
    if args.seed is not None:
        forwarded += ['--seed', str(args.seed)]
    if args.rcwa_python is not None:
        forwarded += ['--rcwa-python', args.rcwa_python]
    if args.skip_rcwa:
        forwarded.append('--skip-rcwa')
    previous = workflow.CASE_PATH
    try:
        workflow.CASE_PATH = path
        return workflow.main(forwarded)
    finally:
        workflow.CASE_PATH = previous


if __name__ == '__main__':
    main()
