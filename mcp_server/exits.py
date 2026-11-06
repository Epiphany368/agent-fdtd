"""The project's shared exit-status vocabulary.

The launcher forwards a driver's exit status; it does not choose it.  ``ok`` is
therefore true only for ``0``.  Codes that describe a lawful scientific outcome
(for example ``57``, a finding) are still ``ok: false`` -- they are not success,
and calling them success is exactly the defect this server exists to prevent.
"""
# Source: README.md, section "Exit statuses you will meet".
_EXITS = {
    0: {
        "meaning": "the criteria that command carries hold",
        "kind": "success",
        "source": "README.md",
    },
    53: {
        "meaning": "a pre-declared stop condition fired; a lawful scientific outcome, not a crash",
        "kind": "stop_condition",
        "source": "README.md",
    },
    54: {
        "meaning": "every criterion is true and the verdict is partial",
        "kind": "partial",
        "source": "README.md",
    },
    55: {
        "meaning": "a declared assumption cannot be relaxed; the run stopped rather than approximated",
        "kind": "assumption",
        "source": "README.md",
    },
    56: {
        "meaning": "a control case did not reproduce its inputs; the comparison is invalid",
        "kind": "control",
        "source": "README.md",
    },
    57: {
        "meaning": "the measured quantity violates its threshold -- a finding, not a crash",
        "kind": "finding",
        "source": "README.md",
    },
    58: {
        "meaning": "a realised (not requested) bound was exceeded",
        "kind": "bound_exceeded",
        "source": "README.md",
    },
    59: {
        "meaning": "out of memory; the run stops and reports and never rewrites solver source",
        "kind": "out_of_memory",
        "source": "README.md",
    },
    61: {
        "meaning": "a pre-declared criterion was violated",
        "kind": "criterion_violated",
        "source": "README.md",
    },
    62: {
        "meaning": "a negative control is missing or vacuous",
        "kind": "negative_control",
        "source": "README.md",
    },
    63: {
        "meaning": "the run carries no criteria block at all",
        "kind": "no_criteria",
        "source": "README.md",
    },
    64: {
        "meaning": "usage error; this is not an executed run",
        "kind": "usage",
        "source": "README.md",
    },
    65: {
        "meaning": "pre-flight refusal; this is not an executed run",
        "kind": "preflight",
        "source": "README.md",
    },
    66: {
        "meaning": "post-flight notary verification failed: telemetry tamper, passivity violation, unverified decay, or cross-solver discrepancy exceeded",
        "kind": "postflight_notary_failed",
        "source": "checks/post_notary.py",
    },
    69: {
        "meaning": "interpreter unavailable; this is not an executed run",
        "kind": "interpreter_unavailable",
        "source": "README.md",
    },
    70: {
        "meaning": "the launcher's own status when a driver exited 0 but the run did not complete",
        "kind": "launcher_incomplete",
        "source": "tools/launch.py",
    },
}

# Codes that mean the launcher never started a simulation.  They are legal
# process statuses, but they are not an executed run.
NOT_EXECUTED = frozenset({64, 65, 69})


def explain(code):
    """Return the entry for ``code``.

    An unknown code is reported as unknown.  The server does not guess a
    meaning for a code that is not in the shared vocabulary.
    """
    try:
        number = int(code)
    except (TypeError, ValueError):
        return {
            "ok": False,
            "known": False,
            "code": code,
            "kind": "unknown",
            "meaning": "not an integer exit status; the shared vocabulary has no entry for it",
            "source": None,
        }
    entry = _EXITS.get(number)
    if entry is None:
        return {
            "ok": False,
            "known": False,
            "code": number,
            "kind": "unknown",
            "meaning": "no entry in the shared exit vocabulary; this server does not guess",
            "source": None,
        }
    return {
        "ok": number == 0,
        "known": True,
        "code": number,
        "kind": entry["kind"],
        "meaning": entry["meaning"],
        "source": entry["source"],
    }


def kind(code):
    return explain(code)["kind"]


def ok(code):
    return explain(code)["ok"]


def vocabulary():
    """The whole known vocabulary, ordered by code, for diagnostics."""
    return [explain(code) for code in sorted(_EXITS)]
