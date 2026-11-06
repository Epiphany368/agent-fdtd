"""Test support for LEG 40: an in-process MCP client over stdio, and fixtures.

No network. The only subprocess started by a tool is ``tools/run.sh``; the fake
solver here is created by the test harness so A3 does not need the real solver
or a GPU.
"""
import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "mcptests" / "fixtures"


class ErrLog:
    """A real file (stdio_client needs fileno) with a convenient getvalue()."""

    def __init__(self):
        self._file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self._content = None

    def write(self, text):
        return self._file.write(text)

    def flush(self):
        return self._file.flush()

    def fileno(self):
        return self._file.fileno()

    def getvalue(self):
        if self._content is not None:
            return self._content
        self._file.flush()
        self._file.seek(0)
        return self._file.read()

    def close(self):
        if self._content is None:
            self._content = self.getvalue()
        self._file.close()


@contextlib.asynccontextmanager
async def open_session(root=None, extra_env=None):
    env = {
        "PYTHONPATH": str(ROOT),
        "PY": sys.executable,
    }
    if root is not None:
        env["TORCHFDTD_PLATFORM_ROOT"] = str(root)
    if extra_env:
        env.update(extra_env)
    errlog = ErrLog()
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server"],
        env=env,
        cwd=str(ROOT),
    )
    try:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session, errlog
    finally:
        errlog.close()


def payload(result):
    """Unwrap a FastMCP CallToolResult into the tool's dict."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        if set(structured) == {"result"} and isinstance(structured["result"], dict):
            return structured["result"]
        return structured
    for block in result.content:
        if getattr(block, "type", None) == "text":
            return json.loads(block.text)
    raise AssertionError(f"tool result carries no structured or JSON text payload: {result!r}")


async def call(session, name, arguments=None):
    return payload(await session.call_tool(name, arguments or {}))


def _git(cwd, *args):
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        text=True,
        capture_output=True,
        check=True,
    )


def _fake_solver(path):
    package = path / "torchfdtd"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "# LEG 40 A3 fixture: a solver-shaped directory; the fixture driver never imports it.\n"
    )
    (path / "LICENSE").write_text("MIT fixture licence\n")
    (path / "THIRD_PARTY_NOTICES.txt").write_text("fixture notices\n")
    (path / "pyproject.toml").write_text(
        '[project]\nname = "torchfdtd-fixture"\nversion = "0.0.0"\n'
    )
    _git(path, "init", "-q")
    _git(path, "add", ".")
    _git(
        path,
        "-c",
        "user.name=LEG 40 fixture",
        "-c",
        "user.email=leg40@example.invalid",
        "commit",
        "-q",
        "-m",
        "fixture solver for the exit-57 run",
    )


@contextlib.contextmanager
def isolated_project(with_solver=True):
    """A minimal project root that runs the real launcher and keeps its ledger."""
    directory = Path(tempfile.mkdtemp(prefix="leg40-mcp-"))
    try:
        shutil.copytree(ROOT / "tools", directory / "tools")
        shutil.copytree(
            ROOT / "checks",
            directory / "checks",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (directory / "configs").mkdir()
        shutil.copy2(ROOT / "configs" / "baseline-pin.json", directory / "configs" / "baseline-pin.json")
        for name in ("plan.md", "README.md", "AGENTS.md"):
            source = ROOT / name
            if source.is_file():
                shutil.copy2(source, directory / name)
            else:
                # The public export deliberately withholds plan.md and AGENTS.md;
                # the launcher snapshots them, so the fixture supplies a stand-in.
                (directory / name).write_text(
                    f"# {name}\n\nstand-in fixture: the real {name} is not part of the public tree.\n",
                    encoding="utf-8",
                )
        (directory / "src").mkdir()
        shutil.copy2(FIXTURES / "exit57.py", directory / "src" / "exit57.py")
        if (ROOT / "src" / "declarative_runner.py").is_file():
            shutil.copy2(ROOT / "src" / "declarative_runner.py", directory / "src" / "declarative_runner.py")
        if (ROOT / "src" / "parity.py").is_file():
            shutil.copy2(ROOT / "src" / "parity.py", directory / "src" / "parity.py")
        shutil.copy2(FIXTURES / "exit57.json", directory / "configs" / "exit57.json")
        shutil.copy2(
            FIXTURES / "invalid-config.json",
            directory / "configs" / "invalid-config.json",
        )
        if with_solver:
            upstream_src = ROOT / "src" / "torchfdtd_upstream"
            if upstream_src.exists():
                try:
                    os.symlink(os.path.realpath(upstream_src), directory / "src" / "torchfdtd_upstream")
                except OSError:
                    pass
            real_solver = ROOT / "src" / "torchfdtd"
            if real_solver.exists():
                try:
                    os.symlink(os.path.realpath(real_solver), directory / "src" / "torchfdtd")
                except OSError:
                    shutil.copytree(real_solver, directory / "src" / "torchfdtd")
            else:
                _fake_solver(directory / "src" / "torchfdtd")
        _git(directory, "init", "-q")
        _git(directory, "add", ".")
        _git(
            directory,
            "-c",
            "user.name=LEG 40 fixture",
            "-c",
            "user.email=leg40@example.invalid",
            "commit",
            "-q",
            "-m",
            "isolated project root for the MCP acceptance tests",
        )
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)
