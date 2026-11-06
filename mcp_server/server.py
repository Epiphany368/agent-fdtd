"""The MCP surface: declarative scene building, resume, and notary verification tools (v2).

No tool calls a model. No tool decides acceptance. ``submit_run`` starts
``tools/run.sh`` and returns its process exit status unchanged.
"""
from mcp.server.fastmcp import FastMCP

from . import platform


INSTRUCTIONS = """\
This server drives the torchfdtd-platform. The model proposes; the harness disposes.
Use list_configs to see what can run, validate_config/validate_project before asking for a run,
and submit_run with dry_run=True first. submit_run executes only tools/run.sh and reports
the process exit status. ok is true only for exit 0: 53/54/55/56/57/58/59 are lawful
scientific outcomes but not successes, and 57 is a finding, not a crash. No tool calls
a model and no tool grants acceptance.
"""


def build_server(host="127.0.0.1", port=8787):
    server = FastMCP(
        "torchfdtd-platform-mcp",
        instructions=INSTRUCTIONS,
        host=host,
        port=port,
        log_level="WARNING",
    )

    @server.tool()
    def list_configs() -> dict:
        """The config/driver pairs that exist, each with a one-line purpose.

        Pairs come from the tracked run ledger. Configs with no recorded driver
        are listed separately; no path is invented.
        """
        return platform.list_configs()

    @server.tool()
    def describe_run(config: str) -> dict:
        """What a config would run: driver, solver role, and the leg's criteria.

        Criteria are quoted from the matching LEG section of plan.md, with the
        line number they came from. If no section is found, none are claimed.
        """
        return platform.describe_run(config)

    @server.tool()
    def validate_config(config: str) -> dict:
        """Run the project's existing config gate and return {ok, errors[]}.

        This calls checks/pre_environment.py; it does not restate its rules and
        runs nothing. Machine probes (git, CUDA) are stubbed, because validation
        is about the config.
        """
        return platform.validation.validate_config(config)

    @server.tool()
    async def submit_run(
        config: str,
        dry_run: bool = True,
        resume_from: str | None = None,
    ) -> dict:
        """Run a config, or show the exact command and pre-flight verdict.

        dry_run=True starts nothing. dry_run=False executes only tools/run.sh and
        returns {ok, exit_status, run_id, ledger_row, stderr_tail}. ok is derived
        from the exit status: 57 is ok false, kind "finding". stderr_tail falls
        back to the stdout tail when stderr is empty, because the launcher logs
        there; tail_source names which stream was used.
        resume_from: optional run_id or journal path to resume from a checkpoint.
        """
        import anyio

        return await anyio.to_thread.run_sync(
            platform.submit_run, config, dry_run, resume_from
        )

    @server.tool()
    def read_ledger(limit: int = 20) -> dict:
        """The last N rows of runs/ledger.jsonl, verbatim and in file order."""
        return platform.read_ledger(limit)

    @server.tool()
    def explain_exit(code: int) -> dict:
        """The shared exit-code vocabulary, including "finding, not a crash".

        An unknown code is reported as unknown; this tool does not guess.
        """
        return platform.explain_exit(code)

    @server.tool()
    def create_project(
        name: str | None = None,
        project_id: str | None = None,
        dimension: str = "2d",
        size: list | None = None,
        domain: dict | None = None,
        mesh: float | None = None,
        grid: dict | None = None,
        steps: int | None = None,
        duration_fs: float | None = None,
        wavelength_um: float | None = None,
        wavelength: float | None = None,
        pml_cells: int | None = None,
        background_index: float | None = None,
        materials: list | None = None,
        structures: list | None = None,
        grating_teeth: dict | None = None,
        source: dict | None = None,
        sources: list | None = None,
        monitors: list | None = None,
        poynting_box: dict | None = None,
        task: str | None = None,
        title: str | None = None,
        precision: str | None = None,
        backend: str | None = None,
        boundaries: dict | None = None,
        max_steps: int | None = None,
        stop_at_step: int | None = None,
        resume_stop_at_step: int | None = None,
        tau_box: float | None = None,
        tau_pml: float | None = None,
        decay_floor: float | None = None,
        incident_flux: float | None = None,
        incident_power: float | None = None,
        project: dict | None = None,
    ) -> dict:
        """Declaratively generate a 2D/3D optical simulation project without writing Python driver code.

        Returns {ok, project_id, config_path, project_canonical_sha256, summary}.
        """
        kwargs = {k: v for k, v in locals().items() if v is not None and k != "project"}
        if project and isinstance(project, dict):
            kwargs.update(project)
        return platform.create_project(**kwargs)

    @server.tool()
    def validate_project(
        config: str | None = None,
        config_path: str | None = None,
        project: dict | None = None,
        check_environment: bool = True,
    ) -> dict:
        """Pre-flight validation for proposed declarative project structures against platform schema and physics gates without running GPU simulations.
        """
        return platform.validate_project(
            project=project,
            config=config,
            config_path=config_path,
            check_environment=check_environment,
        )

    @server.tool()
    async def resume_run(
        run_id: str,
        dry_run: bool = True,
        additional_steps: int | None = None,
    ) -> dict:
        """Resume an interrupted simulation from its ResidentRestartJournal checkpoint.

        Automatically resolves config, driver script, and checkpoint state from
        runs/<run_id>. In dry_run=True mode, validates checkpoint contract and
        reports resume step without executing. In dry_run=False mode, starts
        tools/run.sh with --resume <run_id>.
        """
        import anyio

        return await anyio.to_thread.run_sync(
            platform.resume_run, run_id, dry_run, additional_steps
        )

    @server.tool()
    def inspect_breakpoint(run_id: str) -> dict:
        """Inspect the checkpoint journal and breakpoint status of a simulation run.

        Returns contract invariants, completed steps, total steps, dual ping-pong slot
        integrity (slot_a, slot_b), and process lock ownership (active vs stale lock).
        """
        return platform.inspect_breakpoint(run_id)

    @server.tool()
    def get_certificate(run_id: str) -> dict:
        """Read and verify the post-flight notary certificate and run provenance for a run_id.

        Returns {ok, run_id, verified, certified_utc, status, exit_code, tags, metrics,
        extensions, provenance, failure_cause, error, tamper_detected}.
        verified is true ONLY if checks/post_notary.py issued an authentic certificate
        with all physical invariant and telemetry gates satisfied without subsequent tampering.
        """
        return platform.get_certificate(run_id)

    return server
