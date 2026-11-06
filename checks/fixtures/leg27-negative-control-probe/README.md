# LEG 27 negative-control probe root

**Nothing in this directory is a run record.** These files exist only to prove that
checks/test_leg27_evidence_gaps.py can tell "present" from "absent" -- the negative control for
A2 (`plan.md`, LEG 27): a check that fails unconditionally is not a check, and a check whose
"absent" branch is never observed passing cannot be trusted either.

The probe root sits inside `checks/fixtures/`, never under `runs/`, and every file carries
`"leg27_probe": true`. The meta-check's real-tree scan is `ROOT/runs` only; the probe root is a
parameter, and the two can never mix. The readout bodies here match the shapes the real
predicates require (policy strings, case ids, one sibling file each) with marker payloads, no
physics arrays.

- `probe-root/runs/0000-leg27-probe-leg11-only/readout.json` satisfies leg11's predicate (any
  policy outside the two audit ledgers).
- `probe-root/runs/0001-leg27-probe-leg4-ladder/readout.json` satisfies leg4's
  (`leg4-convergence-v1` + case `ladder-interface-ppw40`).
- `probe-root/runs/0002-leg27-probe-leg12-slab/` satisfies leg12's (`leg12-closed-box-v1` +
  case `slab` + `slab-sample-monitors.npz` beside it; the npz holds one marker array, created by
  `np.savez` once at authoring time, no data).
- `probe-root/runs/0003-leg27-probe-leg15-ladder/readout.json` satisfies leg15's
  (`leg15-closure-v1` newest run with usable cases).
