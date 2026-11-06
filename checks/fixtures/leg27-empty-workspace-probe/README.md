# LEG 27 empty-workspace probe root

This directory exists because git cannot track an empty directory, and checks/test_leg27_evidence_gaps.py
needs a controlled root that is a legal "workspace without archived runs" for its A1 mechanism:
an empty probe root plus no declarations must leave every inventoried leg needed-absent and exit 1.

`probe-root/runs/` is intentionally empty of readout files. Nothing in any
`checks/fixtures/` directory is a run record, and no fixture is ever used to satisfy a
verdict about the real tree: the real scan path is `ROOT/runs` only.
