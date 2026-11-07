"""W3C / RO-Crate 1.1 JSON-LD Evidence Export & Screening Fidelity Gating (LEG 55).

Packages experimental scene IR, Yee grid telemetry, driver readouts, post-flight notary certificates,
and scientific claims into standards-compliant RO-Crate 1.1 JSON-LD datasets.
Strictly blocks screening runs (fidelity_level <= 1) from supporting ScientificClaim with FidelityGateError.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Union

ROOT = Path(__file__).resolve().parents[1]

POSITIVE_CLAIM_STATES = {"SUPPORTED", "PASS", "PASSED", "TRUE", "CONFIRMED", "VALIDATED"}


class FidelityGateError(RuntimeError):
    """Raised when a screening run (fidelity_level <= 1) attempts to support a scientific claim."""


def compute_sha256(path: Path) -> str:
    """Computes SHA-256 digest of file at path."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_screening_fidelity(fidelity_level: int, claim_state: str, parameter: str = "") -> None:
    """Enforces that screening simulations (fidelity_level <= 1) cannot support scientific claims."""
    state = str(claim_state).strip().upper()
    if int(fidelity_level) <= 1 and state in POSITIVE_CLAIM_STATES:
        raise FidelityGateError(
            f"Screening run fidelity violation: fidelity_level={fidelity_level} <= 1 is strictly "
            f"prohibited from asserting SUPPORTED scientific claim for parameter '{parameter}'. (tamper_detected)"
        )


class RoCrateExporter:
    """Exports standardized W3C / RO-Crate 1.1 JSON-LD evidence packages with fidelity gating."""

    def __init__(self, run_dir: Path | str, db_path: Optional[Path | str] = None):
        self.run_dir = Path(run_dir).resolve()
        self.db_path = Path(db_path).resolve() if db_path else None
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

    def export(
        self,
        output_dir: Optional[Path | str] = None,
        claim_id: Optional[str] = None,
    ) -> Path:
        """Export RO-Crate package, strictly enforcing fidelity gating.

        Returns:
            Path to the generated ro-crate-metadata.json file.
        Raises:
            FidelityGateError: if fidelity_level <= 1 and any claim is SUPPORTED.
            FileNotFoundError: if essential readout.json is missing.
        """
        readout_path = self.run_dir / "readout.json"
        if not readout_path.is_file():
            raise FileNotFoundError(f"Missing readout.json in {self.run_dir}")

        readout = json.loads(readout_path.read_text(encoding="utf-8"))

        # 1. Fidelity Level Gating
        fidelity_level = int(readout.get("fidelity_level", 2))
        claims = readout.get("scientific_claims", [])

        for claim in claims:
            state = str(claim.get("claimed_state") or claim.get("actual_state", "")).upper()
            param = str(claim.get("parameter", "unknown"))
            validate_screening_fidelity(fidelity_level, state, param)

        # 2. Setup output directory
        target_name = f"ro_crate_{claim_id}" if claim_id else f"ro_crate_{self.run_dir.name}"
        crate_dir = (Path(output_dir) if output_dir else self.run_dir / "ro_crate") / target_name
        crate_dir.mkdir(parents=True, exist_ok=True)

        # 3. Copy files & calculate checksums
        file_entities = []
        target_files = (
            "experiment_ir.json",
            "readout.json",
            "solver_telemetry.json",
            "verification_certificate.json",
        )
        for filename in target_files:
            src = self.run_dir / filename
            if not src.is_file():
                continue
            dst = crate_dir / filename
            shutil.copy2(src, dst)
            sha = compute_sha256(dst)
            file_type = ["File", "DigitalDocument"] if "certificate" in filename else "File"
            file_entities.append({
                "@id": filename,
                "@type": file_type,
                "name": filename,
                "encodingFormat": "application/json",
                "contentSize": dst.stat().st_size,
                "sha256": sha,
            })

        # 4. Assemble W3C / RO-Crate 1.1 JSON-LD Graph
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        graph: List[Dict[str, Any]] = [
            {
                "@id": "ro-crate-metadata.json",
                "@type": "CreativeWork",
                "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
                "about": {"@id": "./"},
            },
            {
                "@id": "./",
                "@type": "Dataset",
                "name": f"Photonic Experiment Evidence Package: {self.run_dir.name}",
                "description": "Evidence package for verified nanophotonic scientific claim with Two-Tier Error Budget and post-flight notary certificate.",
                "datePublished": now_iso,
                "license": "https://creativecommons.org/licenses/by/4.0/",
                "hasPart": [{"@id": fe["@id"]} for fe in file_entities],
            },
        ]
        graph.extend(file_entities)

        # Add claims as Observation entities
        for idx, claim in enumerate(claims):
            c_id = f"#claim-{claim.get('claim_id') or idx+1}"
            dom = claim.get("budget", {}).get("dominant_component") if isinstance(claim.get("budget"), dict) else None
            graph.append({
                "@id": c_id,
                "@type": ["PropertyValue", "Observation"],
                "name": claim.get("parameter", f"param_{idx}"),
                "value": claim.get("nominal_value"),
                "threshold": claim.get("threshold"),
                "claimState": claim.get("actual_state"),
                "assuranceLevel": claim.get("assurance_level"),
                "fidelityLevel": fidelity_level,
                "dominantComponent": dom,
            })

        # Environment & Provenance Entities
        manifest_path = self.run_dir / "manifest.json"
        manifest_data = {}
        if manifest_path.is_file():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        graph.append({
            "@id": "#torchfdtd-platform",
            "@type": "SoftwareApplication",
            "name": "torchfdtd-platform",
            "softwareVersion": "2.0.0",
            "commitSha": manifest_data.get("project_git_head", "unknown"),
            "codeRepository": "https://github.com/nillinjay/torchfdtd-platform",
        })

        graph.append({
            "@id": f"#run-{self.run_dir.name}",
            "@type": "CreateAction",
            "name": "Photonic Simulation Execution & Verification",
            "startTime": manifest_data.get("started_utc", now_iso),
            "endTime": manifest_data.get("finished_utc", now_iso),
            "instrument": {"@type": "Thing", "gpu": manifest_data.get("gpu", "NVIDIA GeForce RTX 5070 Ti")},
            "result": [{"@id": fe["@id"]} for fe in file_entities],
        })

        crate_meta = {
            "@context": "https://w3id.org/ro/crate/1.1/context",
            "@graph": graph,
        }

        meta_path = crate_dir / "ro-crate-metadata.json"
        meta_path.write_text(json.dumps(crate_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return meta_path


def export_ro_crate(
    run_dir: Path | str,
    output_dir: Optional[Path | str] = None,
    claim_id: Optional[str] = None,
) -> Path:
    """Functional convenience wrapper for RoCrateExporter."""
    exporter = RoCrateExporter(run_dir=run_dir)
    return exporter.export(output_dir=output_dir, claim_id=claim_id)
