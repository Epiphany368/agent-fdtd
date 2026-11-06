"""LEG 40 A3 fixture: a driver whose lawful outcome is exit 57 (a finding).

It is not a simulation and it touches no solver. `tools/run.sh` runs it through
the real launcher, so the run directory and ledger row are produced the same way
a real run's are.
"""
import sys


print("leg40-fixture: measured quantity violates its pre-declared threshold (finding)")
sys.exit(57)
