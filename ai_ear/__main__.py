"""CLI entry point for the AI Ear."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from ai_ear.api.server import create_app
from ai_ear.core.config import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai-ear",
        description="The AI Ear — enterprise-grade multi-modal audio intelligence",
    )
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="Start the REST + WebSocket API server")
    serve_p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve_p.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    serve_p.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")

    args = parser.parse_args(argv)

    if args.command == "serve":
        settings = Settings()
        app = create_app(settings)
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=settings.api_workers,
            reload=args.reload,
            log_level="info",
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
