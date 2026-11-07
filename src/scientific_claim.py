r"""Two-Tier Numerical Error Budget & ScientificClaim Falsification Gate (Wave E — LEG 54).

Provides formal classes and zero-trust integrity verification for nanophotonic scientific claims:
1. ErrorComponent:
   - Individual uncertainty component with explicit certification status and provenance.
2. TwoTierErrorBudget:
   - Aggregates components (U_mesh, U_time, U_box, U_subpixel, U_cross, etc.).
   - Computes combined uncertainty U_total = \sqrt{\sum_i U_i^2}.
   - Classifies assurance_level: "certified" (Level B) iff all components are certified;
     otherwise "estimated" (Level A).
   - Identifies dominant uncertainty component k = \arg\max_i (U_i).
3. ScientificClaim:
   - Evaluates 4-state falsification logic:
     * SUPPORTED: Entire interval strictly above threshold (q_low = q_nom - U_total > q_min).
     * REFUTED: Entire interval strictly below threshold (q_high = q_nom + U_total < q_min).
     * INCONCLUSIVE: Uncertainty interval straddles threshold (q_low <= q_min <= q_high).
     * UNSUPPORTED: Missing evidence, non-monotonic convergence, or uncertified Level B requirement.
   - Generates actionable diagnostic prescription targeting argmax_i U_i when INCONCLUSIVE.
4. validate_claim_integrity:
   - Intercepts fraudulent claims, over-claiming, math tampering, and uncertified Level B
     assertions with RuntimeError("... (tamper_detected)").
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


CLAIM_STATE_SUPPORTED = "SUPPORTED"
CLAIM_STATE_REFUTED = "REFUTED"
CLAIM_STATE_INCONCLUSIVE = "INCONCLUSIVE"
CLAIM_STATE_UNSUPPORTED = "UNSUPPORTED"

VALID_CLAIM_STATES = {
    CLAIM_STATE_SUPPORTED,
    CLAIM_STATE_REFUTED,
    CLAIM_STATE_INCONCLUSIVE,
    CLAIM_STATE_UNSUPPORTED,
}

ASSURANCE_LEVEL_ESTIMATED = "estimated"
ASSURANCE_LEVEL_CERTIFIED = "certified"

VALID_ASSURANCE_LEVELS = {
    ASSURANCE_LEVEL_ESTIMATED,
    ASSURANCE_LEVEL_CERTIFIED,
}


@dataclass
class ErrorComponent:
    """Individual numerical or physical uncertainty component.

    Attributes:
        name: Identifier (e.g. 'U_mesh', 'U_time', 'U_box', 'U_cross', 'U_subpixel').
        value: Non-negative uncertainty magnitude (half-width or 1-sigma).
        certified: True if supported by analytic proof or independent cross-solver envelope;
                   False if an empirical extrapolation/residual estimate.
        method: Methodology used to compute component.
        provenance: Metadata detailing proof theorem, solver versions, convergence order, etc.
    """

    name: str
    value: float
    certified: bool = False
    method: str = "empirical_estimate"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"ErrorComponent name must be non-empty string, got {self.name!r}")
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool) or not math.isfinite(self.value):
            raise ValueError(f"ErrorComponent value must be finite float, got {self.value!r}")
        if self.value < 0.0:
            raise ValueError(f"ErrorComponent value must be non-negative, got {self.value}")
        self.value = float(self.value)
        if not isinstance(self.certified, bool):
            raise ValueError(f"ErrorComponent certified must be boolean, got {type(self.certified)}: {self.certified!r}")
        if not isinstance(self.provenance, dict):
            raise ValueError(f"ErrorComponent provenance must be dict, got {type(self.provenance)}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "certified": self.certified,
            "method": self.method,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ErrorComponent:
        cert = data.get("certified", False)
        if not isinstance(cert, bool):
            raise RuntimeError(
                f"ScientificClaim integrity violation: component {data.get('name')} certified must be boolean (tamper_detected)"
            )
        return cls(
            name=str(data["name"]),
            value=float(data["value"]),
            certified=cert,
            method=str(data.get("method", "empirical_estimate")),
            provenance=dict(data.get("provenance", {})),
        )


@dataclass
class TwoTierErrorBudget:
    """Two-Tier Numerical Error Budget system.

    Level A ("estimated"):
        Based on empirical monotonic grid refinement (Richardson/GCI U_mesh),
        temporal ringdown decay (U_time), and Poynting box closure (U_box).
        Only supports engineering claims under assurance_level="estimated".

    Level B ("certified"):
        Requires Level A plus independent cross-solver envelope (U_cross from RCWA/Meep/analytic)
        or explicit analytic proof bounds for all constituent components.
    """

    components: List[ErrorComponent] = field(default_factory=list)
    combination_method: str = "rss"  # 'rss' (root-sum-square) or 'linear'

    def __post_init__(self) -> None:
        cleaned: List[ErrorComponent] = []
        for c in self.components:
            if isinstance(c, ErrorComponent):
                cleaned.append(c)
            elif isinstance(c, dict):
                cleaned.append(ErrorComponent.from_dict(c))
            else:
                raise TypeError(f"Invalid component type in TwoTierErrorBudget: {type(c)}")
        self.components = cleaned

    def add_component(
        self,
        name: str,
        value: float,
        certified: bool = False,
        method: str = "empirical_estimate",
        provenance: Optional[Dict[str, Any]] = None,
    ) -> ErrorComponent:
        comp = ErrorComponent(
            name=name,
            value=value,
            certified=certified,
            method=method,
            provenance=provenance or {},
        )
        self.components.append(comp)
        return comp

    @property
    def total_uncertainty(self) -> float:
        """Combined total uncertainty U_total."""
        if not self.components:
            return 0.0
        if self.combination_method == "linear":
            return float(sum(c.value for c in self.components))
        # Default: RSS combination U_total = \sqrt{\sum_i U_i^2}
        return float(math.sqrt(sum(c.value * c.value for c in self.components)))

    @property
    def assurance_level(self) -> str:
        """Determines whether the budget meets Level B ('certified') or Level A ('estimated').

        Strict invariant: assurance_level is 'certified' IF AND ONLY IF all components have
        certified=True and at least one component is present.
        """
        if not self.components:
            return ASSURANCE_LEVEL_ESTIMATED
        if all(c.certified for c in self.components):
            return ASSURANCE_LEVEL_CERTIFIED
        return ASSURANCE_LEVEL_ESTIMATED

    @property
    def is_certified(self) -> bool:
        return self.assurance_level == ASSURANCE_LEVEL_CERTIFIED

    @property
    def dominant_component(self) -> Optional[ErrorComponent]:
        """Identify the uncertainty component with maximum magnitude: k = argmax_i U_i."""
        if not self.components:
            return None
        return max(self.components, key=lambda c: c.value)

    def component_percentages(self) -> Dict[str, float]:
        """Return percentage contribution of each component to total variance U_total^2."""
        u_tot = self.total_uncertainty
        if u_tot <= 0.0:
            return {c.name: 0.0 for c in self.components}
        u_tot_sq = u_tot * u_tot
        return {c.name: (c.value * c.value / u_tot_sq) * 100.0 for c in self.components}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": [c.to_dict() for c in self.components],
            "combination_method": self.combination_method,
            "total_uncertainty": self.total_uncertainty,
            "assurance_level": self.assurance_level,
            "dominant_component": self.dominant_component.name if self.dominant_component else None,
            "component_percentages": self.component_percentages(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TwoTierErrorBudget:
        comps = [ErrorComponent.from_dict(c) for c in data.get("components", [])]
        method = str(data.get("combination_method", "rss"))
        return cls(components=comps, combination_method=method)


@dataclass
class ScientificClaim:
    r"""4-State Scientific Claim Evaluation and Falsification Gate.

    Fields:
        parameter: Physical observable name (e.g. 'coupling_efficiency_eta').
        nominal_value: Point estimate \hat{q}.
        budget: Associated TwoTierErrorBudget.
        threshold: Decision threshold q_min (or target value).
        claim_type: 'greater_than' (default) or 'less_than'.
        required_assurance_level: Required assurance level ('estimated' or 'certified').
        claimed_state: State claimed by driver/user (None until declared).
        actual_state: Mathematically derived state (SUPPORTED, REFUTED, INCONCLUSIVE, UNSUPPORTED).
        diagnostic_prescription: Actionable prescription targeting argmax_i U_i when INCONCLUSIVE.
        provenance: Context metadata.
    """

    parameter: str
    nominal_value: float
    budget: TwoTierErrorBudget
    threshold: float
    claim_type: str = "greater_than"
    required_assurance_level: str = ASSURANCE_LEVEL_ESTIMATED
    claimed_state: Optional[str] = None
    actual_state: str = ""
    diagnostic_prescription: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.budget, TwoTierErrorBudget):
            if isinstance(self.budget, dict):
                self.budget = TwoTierErrorBudget.from_dict(self.budget)
            else:
                raise TypeError(f"budget must be TwoTierErrorBudget, got {type(self.budget)}")
        self.nominal_value = float(self.nominal_value)
        self.threshold = float(self.threshold)
        if self.claim_type not in ("greater_than", "less_than"):
            raise ValueError(f"Unsupported claim_type: {self.claim_type!r}")
        if self.required_assurance_level not in VALID_ASSURANCE_LEVELS:
            raise ValueError(f"Unsupported required_assurance_level: {self.required_assurance_level!r}")

        # Derive actual state if not yet derived
        if not self.actual_state:
            self.evaluate()

    @property
    def interval(self) -> Tuple[float, float]:
        r"""Uncertainty interval [q_low, q_high] = [\hat{q} - U_total, \hat{q} + U_total]."""
        u_tot = self.budget.total_uncertainty
        return (self.nominal_value - u_tot, self.nominal_value + u_tot)

    @property
    def assurance_level(self) -> str:
        return self.budget.assurance_level

    def evaluate(self) -> str:
        r"""Evaluate 4-state mathematical logic:

        For 'greater_than':
        - q_low > q_min:
            If required_assurance_level == 'certified' and budget is only 'estimated':
                Cannot certify Level B -> INCONCLUSIVE (or UNSUPPORTED for Level B claim).
            Else:
                SUPPORTED
        - q_high < q_min:
            REFUTED (entire interval falls below threshold).
        - q_low <= q_min <= q_high:
            INCONCLUSIVE (interval straddles threshold).
            Emits actionable prescription for argmax_i U_i.
        """
        q_low, q_high = self.interval
        q_min = self.threshold

        if self.claim_type == "greater_than":
            if q_low > q_min:
                # Check Level B requirement
                if self.required_assurance_level == ASSURANCE_LEVEL_CERTIFIED and self.budget.assurance_level != ASSURANCE_LEVEL_CERTIFIED:
                    self.actual_state = CLAIM_STATE_INCONCLUSIVE
                    self.diagnostic_prescription = (
                        "Level B certification required: uncertainty interval q_low > q_min holds under empirical budget, "
                        "but assurance_level is 'estimated'. Independent cross-solver or analytic proof is required to certify."
                    )
                else:
                    self.actual_state = CLAIM_STATE_SUPPORTED
                    self.diagnostic_prescription = None
            elif q_high < q_min:
                self.actual_state = CLAIM_STATE_REFUTED
                self.diagnostic_prescription = None
            else:
                self.actual_state = CLAIM_STATE_INCONCLUSIVE
                self.diagnostic_prescription = self._build_diagnostic_prescription()
        elif self.claim_type == "less_than":
            if q_high < q_min:
                if self.required_assurance_level == ASSURANCE_LEVEL_CERTIFIED and self.budget.assurance_level != ASSURANCE_LEVEL_CERTIFIED:
                    self.actual_state = CLAIM_STATE_INCONCLUSIVE
                    self.diagnostic_prescription = (
                        "Level B certification required: uncertainty interval q_high < q_max holds under empirical budget, "
                        "but assurance_level is 'estimated'. Independent cross-solver or analytic proof is required to certify."
                    )
                else:
                    self.actual_state = CLAIM_STATE_SUPPORTED
                    self.diagnostic_prescription = None
            elif q_low > q_min:
                self.actual_state = CLAIM_STATE_REFUTED
                self.diagnostic_prescription = None
            else:
                self.actual_state = CLAIM_STATE_INCONCLUSIVE
                self.diagnostic_prescription = self._build_diagnostic_prescription()

        return self.actual_state

    def _build_diagnostic_prescription(self) -> str:
        """Generate targeted prescription for dominant component k = argmax_i U_i."""
        dom = self.budget.dominant_component
        if dom is None or dom.value <= 0.0:
            return "Uncertainty budget is unpopulated or zero; cannot determine dominant error component."

        pcts = self.budget.component_percentages()
        dom_pct = pcts.get(dom.name, 0.0)
        dom_lower = dom.name.lower()

        if any(w in dom_lower for w in ("grid", "mesh", "yee", "dx")):
            action = (
                f"Refine Yee mesh resolution or expand convergence ladder rungs: "
                f"grid uncertainty '{dom.name}' accounts for {dom_pct:.1f}% of uncertainty budget (U = {dom.value:.4e})."
            )
        elif any(w in dom_lower for w in ("time", "ringdown", "temporal", "decay")):
            action = (
                f"Extend simulation duration or lower auto-shutoff threshold: "
                f"temporal ringdown uncertainty '{dom.name}' accounts for {dom_pct:.1f}% of uncertainty budget (U = {dom.value:.4e})."
            )
        elif any(w in dom_lower for w in ("box", "poynting", "closure")):
            action = (
                f"Enlarge Poynting box domain margin or check monitor alignment: "
                f"box closure residual '{dom.name}' accounts for {dom_pct:.1f}% of uncertainty budget (U = {dom.value:.4e})."
            )
        elif any(w in dom_lower for w in ("pml", "boundary", "cpml", "reflection")):
            action = (
                f"Increase CPML layer thickness or optimize polynomial profile: "
                f"boundary reflection '{dom.name}' accounts for {dom_pct:.1f}% of uncertainty budget (U = {dom.value:.4e})."
            )
        elif any(w in dom_lower for w in ("subpixel", "etch", "kink", "bias")):
            action = (
                f"Increase subpixel integration quadrature or refine boundary etch-bias discretization: "
                f"subpixel uncertainty '{dom.name}' accounts for {dom_pct:.1f}% of uncertainty budget (U = {dom.value:.4e})."
            )
        elif any(w in dom_lower for w in ("cross", "solver", "reconciliation")):
            action = (
                f"Execute high-order cross-solver reconciliation against reference solver: "
                f"cross-solver discrepancy '{dom.name}' accounts for {dom_pct:.1f}% of uncertainty budget (U = {dom.value:.4e})."
            )
        else:
            action = (
                f"Target dominant error component '{dom.name}' (U = {dom.value:.4e}, {dom_pct:.1f}% of budget) "
                f"for resolution refinement or tighter convergence bounds."
            )

        return action

    def to_dict(self) -> Dict[str, Any]:
        q_low, q_high = self.interval
        return {
            "parameter": self.parameter,
            "nominal_value": self.nominal_value,
            "threshold": self.threshold,
            "claim_type": self.claim_type,
            "required_assurance_level": self.required_assurance_level,
            "claimed_state": self.claimed_state,
            "actual_state": self.actual_state,
            "interval": [q_low, q_high],
            "total_uncertainty": self.budget.total_uncertainty,
            "assurance_level": self.budget.assurance_level,
            "diagnostic_prescription": self.diagnostic_prescription,
            "budget": self.budget.to_dict(),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScientificClaim:
        budget_data = data.get("budget", {})
        if isinstance(budget_data, TwoTierErrorBudget):
            budget = budget_data
        else:
            budget = TwoTierErrorBudget.from_dict(budget_data)

        claim = cls(
            parameter=str(data["parameter"]),
            nominal_value=float(data["nominal_value"]),
            budget=budget,
            threshold=float(data["threshold"]),
            claim_type=str(data.get("claim_type", "greater_than")),
            required_assurance_level=str(data.get("required_assurance_level", ASSURANCE_LEVEL_ESTIMATED)),
            claimed_state=data.get("claimed_state"),
            actual_state=str(data.get("actual_state", "")),
            diagnostic_prescription=data.get("diagnostic_prescription"),
            provenance=dict(data.get("provenance", {})),
        )
        return claim


def validate_claim_integrity(claim_or_dict: Union[ScientificClaim, Dict[str, Any]]) -> bool:
    r"""Zero-Trust Mathematical Integrity Validator for Scientific Claims.

    Enforces:
    1. Budget existence: Non-empty uncertainty budget with positive components (no zero-budget cheating).
    2. RSS Math Consistency: U_total == \sqrt{\sum U_i^2} within 1e-6.
    3. Assurance Level Gating:
       Cannot assert assurance_level="certified" unless ALL components are verified certified.
    4. 4-State Truth Consistency:
       - If q_low <= q_min <= q_high (INCONCLUSIVE), claiming SUPPORTED raises
         RuntimeError("... (tamper_detected)").
       - If q_high < q_min (REFUTED), claiming SUPPORTED raises
         RuntimeError("... (tamper_detected)").
       - Declared actual_state must exactly match mathematically derived state.
       - INCONCLUSIVE claims must provide non-empty diagnostic prescription naming dominant component.

    Returns:
        True if all mathematical and integrity invariants are strictly satisfied.
    Raises:
        RuntimeError with (tamper_detected) tag on any integrity violation.
    """
    if isinstance(claim_or_dict, ScientificClaim):
        claim_dict = claim_or_dict.to_dict()
    elif isinstance(claim_or_dict, dict):
        claim_dict = claim_or_dict
    else:
        raise RuntimeError(f"Invalid claim object type: {type(claim_or_dict)} (tamper_detected)")

    # 1. Budget extraction and structure validation
    budget_data = claim_dict.get("budget")
    if not isinstance(budget_data, dict):
        raise RuntimeError("ScientificClaim integrity violation: missing or invalid budget dictionary (tamper_detected)")

    components = budget_data.get("components", [])
    if not isinstance(components, list) or len(components) == 0:
        raise RuntimeError("ScientificClaim integrity violation: omitted or zero uncertainty budget (tamper_detected)")

    comp_vals = []
    comp_certified = []
    for idx, comp in enumerate(components):
        if not isinstance(comp, dict) or "value" not in comp:
            raise RuntimeError(f"ScientificClaim integrity violation: corrupted component {idx} (tamper_detected)")
        val = comp["value"]
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val) or val < 0.0:
            raise RuntimeError(f"ScientificClaim integrity violation: invalid component value {val} (tamper_detected)")
        comp_vals.append(float(val))
        c_cert = comp.get("certified", False)
        if not isinstance(c_cert, bool):
            raise RuntimeError(
                f"ScientificClaim integrity violation: component {idx} certified must be boolean (tamper_detected)"
            )
        if c_cert:
            prov = comp.get("provenance")
            if not isinstance(prov, dict) or not prov or not any(k in prov for k in ("proof", "solver", "theorem", "reference_solver", "envelope", "analytic")):
                raise RuntimeError(
                    f"ScientificClaim integrity violation: component {idx} claims certified=True but lacks proof provenance (tamper_detected)"
                )
        comp_certified.append(c_cert)

    # 2. Mathematical RSS Consistency Check
    combination_method = budget_data.get("combination_method", "rss")
    if combination_method == "linear":
        expected_total = sum(comp_vals)
    else:
        expected_total = math.sqrt(sum(v * v for v in comp_vals))

    reported_total = budget_data.get("total_uncertainty")
    if reported_total is None or not isinstance(reported_total, (int, float)) or isinstance(reported_total, bool) or not math.isfinite(reported_total):
        raise RuntimeError("ScientificClaim integrity violation: missing or non-finite total_uncertainty (tamper_detected)")
    reported_total = float(reported_total)

    if not math.isfinite(expected_total) or math.isnan(expected_total) or math.isinf(expected_total):
        raise RuntimeError("ScientificClaim integrity violation: non-finite combined uncertainty calculation (tamper_detected)")

    if abs(reported_total - expected_total) > 1e-6:
        raise RuntimeError(
            f"ScientificClaim integrity violation: inconsistent combined uncertainty math: "
            f"reported={reported_total:.6e} vs expected={expected_total:.6e} (tamper_detected)"
        )

    if reported_total <= 0.0:
        raise RuntimeError("ScientificClaim integrity violation: non-positive total uncertainty budget (tamper_detected)")

    # 3. Assurance Level Gating
    budget_assurance = budget_data.get("assurance_level", ASSURANCE_LEVEL_ESTIMATED)
    claim_assurance = claim_dict.get("assurance_level", budget_assurance)
    if budget_assurance not in VALID_ASSURANCE_LEVELS or claim_assurance not in VALID_ASSURANCE_LEVELS:
        raise RuntimeError(f"ScientificClaim integrity violation: invalid assurance_level {claim_assurance!r} (tamper_detected)")

    all_certified = all(comp_certified)
    if budget_assurance == ASSURANCE_LEVEL_CERTIFIED and not all_certified:
        raise RuntimeError(
            "ScientificClaim integrity violation: budget claims 'certified' assurance level but has uncertified components (tamper_detected)"
        )
    if claim_assurance == ASSURANCE_LEVEL_CERTIFIED and not all_certified:
        raise RuntimeError(
            "ScientificClaim integrity violation: claim claims 'certified' assurance level with uncertified components (tamper_detected)"
        )

    # 4. Mathematical 4-State Verification
    nominal_val = claim_dict.get("nominal_value")
    threshold = claim_dict.get("threshold")
    claim_type = claim_dict.get("claim_type", "greater_than")
    required_assurance = claim_dict.get("required_assurance_level", ASSURANCE_LEVEL_ESTIMATED)

    if not isinstance(nominal_val, (int, float)) or not isinstance(threshold, (int, float)):
        raise RuntimeError("ScientificClaim integrity violation: missing nominal_value or threshold (tamper_detected)")
    nominal_val = float(nominal_val)
    threshold = float(threshold)

    q_low = nominal_val - reported_total
    q_high = nominal_val + reported_total

    # Derive ground-truth mathematical state
    if claim_type == "greater_than":
        if q_low > threshold:
            if required_assurance == ASSURANCE_LEVEL_CERTIFIED and not all_certified:
                math_state = CLAIM_STATE_INCONCLUSIVE
            else:
                math_state = CLAIM_STATE_SUPPORTED
        elif q_high < threshold:
            math_state = CLAIM_STATE_REFUTED
        else:
            math_state = CLAIM_STATE_INCONCLUSIVE
    elif claim_type == "less_than":
        if q_high < threshold:
            if required_assurance == ASSURANCE_LEVEL_CERTIFIED and not all_certified:
                math_state = CLAIM_STATE_INCONCLUSIVE
            else:
                math_state = CLAIM_STATE_SUPPORTED
        elif q_low > threshold:
            math_state = CLAIM_STATE_REFUTED
        else:
            math_state = CLAIM_STATE_INCONCLUSIVE
    else:
        raise RuntimeError(f"ScientificClaim integrity violation: unsupported claim_type {claim_type!r} (tamper_detected)")

    # Check claimed_state if present
    raw_claimed = claim_dict.get("claimed_state") or claim_dict.get("claim_status")
    claimed_state = str(raw_claimed).strip().upper() if raw_claimed is not None else None
    positive_synonyms = ("SUPPORTED", "PASS", "PASSED", "CONFIRMED", "TRUE", "VALIDATED")
    if claimed_state in positive_synonyms:
        if math_state == CLAIM_STATE_INCONCLUSIVE:
            raise RuntimeError(
                f"ScientificClaim integrity violation: falsely claimed {claimed_state} when interval "
                f"[{q_low:.4f}, {q_high:.4f}] is INCONCLUSIVE against threshold {threshold:.4f} (tamper_detected)"
            )
        if math_state == CLAIM_STATE_REFUTED:
            raise RuntimeError(
                f"ScientificClaim integrity violation: falsely claimed {claimed_state} when interval "
                f"[{q_low:.4f}, {q_high:.4f}] is REFUTED against threshold {threshold:.4f} (tamper_detected)"
            )

    # Check actual_state declared in claim_dict
    declared_actual = claim_dict.get("actual_state")
    if declared_actual:
        if declared_actual != math_state:
            raise RuntimeError(
                f"ScientificClaim integrity violation: declared actual_state '{declared_actual}' contradicts "
                f"mathematical evaluation '{math_state}' on interval [{q_low:.4f}, {q_high:.4f}] (tamper_detected)"
            )

    # Check diagnostic prescription when inconclusive
    if math_state == CLAIM_STATE_INCONCLUSIVE:
        prescription = claim_dict.get("diagnostic_prescription")
        if not prescription or not isinstance(prescription, str) or not prescription.strip():
            raise RuntimeError(
                "ScientificClaim integrity violation: missing diagnostic prescription for INCONCLUSIVE claim (tamper_detected)"
            )
        # Verify prescription references the dominant component
        dominant_comp_name = budget_data.get("dominant_component")
        if not dominant_comp_name:
            dom = max(components, key=lambda c: c["value"])
            dominant_comp_name = dom["name"]
        if dominant_comp_name not in prescription and "Level B" not in prescription:
            raise RuntimeError(
                f"ScientificClaim integrity violation: diagnostic prescription does not target dominant component "
                f"'{dominant_comp_name}' (tamper_detected)"
            )

    return True
