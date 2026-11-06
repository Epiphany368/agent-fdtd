"""Fixture integrity and anti-omission gate.

Ensures that all test fixtures across the platform (checks/fixtures/ and mcptests/fixtures/)
are present on disk, non-empty, byte-identical to their pinned cryptographic SHA-256 hashes,
clean in git status, and syntactically parseable.

Includes explicit negative control tests to satisfy AGENTS.md contract:
"The negative-control rule survives at both granularities -- a gate whose failure
mode has never been produced is a description, not a gate."
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

def _find_repo_root():
    cur = Path(__file__).resolve().parent
    while cur != cur.parent:
        if (cur / '.git').exists() or ((cur / 'checks').exists() and (cur / 'tools').exists()):
            return cur
        cur = cur.parent
    return Path(__file__).resolve().parents[1]

ROOT = _find_repo_root()

# Pinned cryptographic inventory of all platform test fixtures.
# Any addition, deletion, or modification of fixtures must be registered here.
# Sizes and SHA-256 hashes are anchored to Git HEAD blobs (canonical LF line endings).
FIXTURE_MANIFEST = {
    # --- checks/fixtures/ (LEG 27 evidence gap meta-check fixtures) ---
    'checks/fixtures/leg27-empty-workspace-probe/README.md': {
        'canonical_size': 571,
        'sha256': '206717b23e68cd0ce1254df65fb718f1272416a963748c914321d01723196e73',
        'format': 'text',
    },
    'checks/fixtures/leg27-negative-control-probe/README.md': {
        'canonical_size': 1409,
        'sha256': '35afc51efa6b76df5388e5592a2887bf8b90a124265cf8ad280d320096bfe89e',
        'format': 'text',
    },
    'checks/fixtures/leg27-negative-control-probe/probe-root/runs/0000-leg27-probe-leg11-only/readout.json': {
        'canonical_size': 226,
        'sha256': '5fbed451060956777efe494c27d02659f627a5b6b1ae26e7020baf6f7dc286df',
        'format': 'json',
    },
    'checks/fixtures/leg27-negative-control-probe/probe-root/runs/0001-leg27-probe-leg4-ladder/readout.json': {
        'canonical_size': 303,
        'sha256': 'cc03aceb3b7cec47229656611608c3d92a5d913452cb136d8fba4305407c9ab9',
        'format': 'json',
    },
    'checks/fixtures/leg27-negative-control-probe/probe-root/runs/0002-leg27-probe-leg12-slab/readout.json': {
        'canonical_size': 285,
        'sha256': '00850efa73a3da22be93ffdecf24c60b6da7b5d43fbd15406328134cf7e58d40',
        'format': 'json',
    },
    'checks/fixtures/leg27-negative-control-probe/probe-root/runs/0002-leg27-probe-leg12-slab/slab-sample-monitors.npz': {
        'canonical_size': 322,
        'sha256': '2b056ea8a72a405b06b57cb245d3632e5b64e1f84e566d612381ab25c8a038f3',
        'format': 'npz',
    },
    'checks/fixtures/leg27-negative-control-probe/probe-root/runs/0003-leg27-probe-leg15-ladder/readout.json': {
        'canonical_size': 297,
        'sha256': 'daf771e762da8b81e24c4c4258fcc9ee4a7e26e46056feb787e3429b580714d6',
        'format': 'json',
    },

    # --- mcptests/fixtures/ (LEG 40 MCP server test fixtures) ---
    'mcptests/fixtures/README.md': {
        'canonical_size': 717,
        'sha256': '5bd2b7e5b0fafe2bff74556ededd96d5b22b6a7596b5f8cf66ef297d523617c8',
        'format': 'text',
    },
    'mcptests/fixtures/exit57.json': {
        'canonical_size': 232,
        'sha256': '54ef246eccad266d8924f0a4842340678a1ddaaf299517bc2c0b336e39c045f7',
        'format': 'json',
    },
    'mcptests/fixtures/exit57.py': {
        'canonical_size': 374,
        'sha256': '35791fecee5bd0b630b99016f1a35ce5185bbda5caf40251f7c221860383b299',
        'format': 'python',
    },
    'mcptests/fixtures/invalid-config.json': {
        'canonical_size': 22,
        'sha256': '307b47e5939d13f5e8b5fa99a855ad156dadb78b7f1061198ceeada448a6ec94',
        'format': 'json',
    },
}


def verify_manifest(root: Path, manifest: dict = FIXTURE_MANIFEST) -> list:
    """Verify all fixtures against manifest. Returns list of error strings."""
    errors = []
    for rel_path, spec in sorted(manifest.items()):
        path = root / rel_path
        if not path.is_file():
            errors.append(f"MISSING: fixture file does not exist: {rel_path}")
            continue

        raw = path.read_bytes()
        if len(raw) == 0:
            errors.append(f"EMPTY: fixture file has 0 bytes: {rel_path}")
            continue

        fmt = spec.get('format')
        # Binary formats must match exact byte size and hash
        if fmt == 'npz':
            canonical_bytes = raw
        else:
            # Text/JSON/Python files: normalize CRLF -> LF for cross-platform comparison
            canonical_bytes = raw.replace(b'\r\n', b'\n')

        if len(canonical_bytes) != spec['canonical_size']:
            errors.append(
                f"SIZE_MISMATCH: {rel_path} canonical size {len(canonical_bytes)} != expected {spec['canonical_size']}"
            )

        digest = hashlib.sha256(canonical_bytes).hexdigest()
        if digest != spec['sha256']:
            errors.append(
                f"HASH_MISMATCH: {rel_path} sha256 {digest} != expected {spec['sha256']}"
            )

        # Content parseability verification
        if fmt == 'json':
            try:
                json.loads(canonical_bytes.decode('utf-8'))
            except Exception as exc:
                errors.append(f"PARSE_ERROR: {rel_path} invalid JSON: {exc}")
        elif fmt == 'python':
            try:
                ast.parse(canonical_bytes.decode('utf-8'), filename=str(path))
            except Exception as exc:
                errors.append(f"PARSE_ERROR: {rel_path} invalid Python syntax: {exc}")
        elif fmt == 'npz':
            try:
                with zipfile.ZipFile(path, 'r') as zf:
                    names = zf.namelist()
                    if not names:
                        errors.append(f"PARSE_ERROR: {rel_path} is an empty zip/npz archive")
            except Exception as exc:
                errors.append(f"PARSE_ERROR: {rel_path} invalid zip/npz format: {exc}")

    return errors


class FixtureIntegrityTests(unittest.TestCase):
    def test_all_fixtures_present_and_match_manifest(self):
        """Every pinned fixture must be present, non-empty, and cryptographically bit-identical."""
        errors = verify_manifest(ROOT)
        self.assertFalse(errors, "Fixture integrity failures detected:\n" + "\n".join(errors))

    def test_git_status_fixtures_clean(self):
        """No fixture directory should have unstaged deletions, modifications, or staged changes."""
        try:
            # Query git with core.fileMode=false to prevent NTFS permission false positives
            res = subprocess.run(
                ['git', '-C', str(ROOT), '-c', 'core.fileMode=false', 'status', '--porcelain', 'checks/fixtures/', 'mcptests/fixtures/'],
                capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("git command unavailable or not a git checkout")
        dirty_lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        self.assertFalse(
            dirty_lines,
            "Git working tree has uncommitted fixture changes:\n" + "\n".join(dirty_lines)
        )

    def test_git_tracked_fixtures_match_manifest(self):
        """Every fixture tracked by git in checks/fixtures and mcptests/fixtures must be in manifest."""
        try:
            res = subprocess.run(
                ['git', '-C', str(ROOT), 'ls-files', 'checks/fixtures/', 'mcptests/fixtures/'],
                capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("git command unavailable or not a git checkout")
        tracked_files = set(line.strip().replace('\\', '/') for line in res.stdout.splitlines() if line.strip())
        manifest_files = set(FIXTURE_MANIFEST.keys())

        untracked_in_manifest = manifest_files - tracked_files
        self.assertFalse(
            untracked_in_manifest,
            f"Manifest lists fixtures not tracked by git: {untracked_in_manifest}"
        )
        unmanifested_in_git = tracked_files - manifest_files
        self.assertFalse(
            unmanifested_in_git,
            f"Git tracks fixtures missing from FIXTURE_MANIFEST: {unmanifested_in_git}"
        )


class FixtureIntegrityNegativeControlTests(unittest.TestCase):
    """Negative controls demonstrating that verify_manifest raises errors when integrity is violated."""

    def test_negative_control_missing_file_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            mock_manifest = {
                'checks/fixtures/missing.json': {
                    'canonical_size': 100,
                    'sha256': '0000000000000000000000000000000000000000000000000000000000000000',
                    'format': 'json',
                }
            }
            errors = verify_manifest(tmp_root, mock_manifest)
            self.assertTrue(errors)
            self.assertTrue(any('MISSING' in err and 'missing.json' in err for err in errors))

    def test_negative_control_empty_file_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            f = tmp_root / 'empty.json'
            f.touch()
            mock_manifest = {
                'empty.json': {
                    'canonical_size': 10,
                    'sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
                    'format': 'json',
                }
            }
            errors = verify_manifest(tmp_root, mock_manifest)
            self.assertTrue(any('EMPTY' in err and 'empty.json' in err for err in errors))

    def test_negative_control_tampered_hash_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            f = tmp_root / 'tampered.json'
            f.write_text('{"tampered": true}\n')
            mock_manifest = {
                'tampered.json': {
                    'canonical_size': len(f.read_bytes().replace(b'\r\n', b'\n')),
                    'sha256': '1111111111111111111111111111111111111111111111111111111111111111',
                    'format': 'json',
                }
            }
            errors = verify_manifest(tmp_root, mock_manifest)
            self.assertTrue(any('HASH_MISMATCH' in err and 'tampered.json' in err for err in errors))

    def test_negative_control_corrupt_json_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            f = tmp_root / 'corrupt.json'
            f.write_bytes(b'{invalid json: 123\n')
            canonical = f.read_bytes().replace(b'\r\n', b'\n')
            digest = hashlib.sha256(canonical).hexdigest()
            mock_manifest = {
                'corrupt.json': {
                    'canonical_size': len(canonical),
                    'sha256': digest,
                    'format': 'json',
                }
            }
            errors = verify_manifest(tmp_root, mock_manifest)
            self.assertTrue(any('PARSE_ERROR' in err and 'corrupt.json' in err for err in errors))


if __name__ == '__main__':
    unittest.main()
