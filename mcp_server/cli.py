"""Console entry point: ``torchfdtd-platform-mcp``.

Default transport is stdio. ``--http`` switches to the MCP streamable HTTP
transport, reachable at ``http://<host>:<port>/mcp``. The default host is
loopback; a remote client needs ``--host 0.0.0.0`` (or a specific interface).
"""
import argparse

from .server import build_server


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="torchfdtd-platform-mcp",
        description="MCP server for the torchfdtd-platform (the exit status decides acceptance)",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="serve the MCP streamable HTTP transport instead of stdio",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP bind address (default 127.0.0.1; use 0.0.0.0 for remote clients)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8787,
        help="HTTP port (default 8787)",
    )
    args = parser.parse_args(argv)

    server = build_server(host=args.host, port=args.port)
    server.run(transport="streamable-http" if args.http else "stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
