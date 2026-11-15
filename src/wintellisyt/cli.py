"""Command-line entry points for local research runs."""

import argparse
import tempfile
from pathlib import Path

from .fixtures import write_fixture
from .pipeline import run, source_manifest

SOURCES = ("kelmarsh", "penmanshiel", "la_haute_borne")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wintellisyt")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run the deterministic synthetic smoke fixture")
    analysis = commands.add_parser("run", help="audit and analyze a local source file")
    commands.add_parser("sources", help="show source provenance coverage")
    comparison = commands.add_parser("transfer", help="score a verified held-out turbine pair")
    for side in ("train", "target"):
        comparison.add_argument(f"--{side}-source", choices=SOURCES, required=True)
        comparison.add_argument(f"--{side}-input", type=Path, required=True)
        comparison.add_argument(f"--{side}-turbine", required=True)
    comparison.add_argument("--output", required=True, type=Path)
    comparison.add_argument("--min-bin-count", type=int, default=5)
    ui_cmd = commands.add_parser("ui", help="launch interactive NiceGUI/Starlette dashboard or export HTML")
    ui_cmd.add_argument("--suite-dir", type=Path, default=Path("artifacts/engineering_suite_2017"))
    ui_cmd.add_argument("--export-only", action="store_true")
    ui_cmd.add_argument("--output", type=Path, default=None)
    ui_cmd.add_argument("--host", default="127.0.0.1")
    ui_cmd.add_argument("--port", type=int, default=8080)
    ui_cmd.add_argument("--backend", choices=["auto", "nicegui", "starlette"], default="auto")
    ui_cmd.add_argument("--show", action="store_true", help="Automatically open the local dashboard URL in the browser.")
    analysis.add_argument("--source", required=True, choices=SOURCES)
    analysis.add_argument("--input", required=True, type=Path)
    for command in (analysis, demo):
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--turbine")
        command.add_argument("--min-bin-count", type=int, default=5)
    args = parser.parse_args(argv)

    try:
        if args.command == "sources":
            for name, info in source_manifest().items():
                print(f"{name}: {info['acquisition']['state']} | {info['source_url']}")
            return 0
        if args.command == "ui":
            from .ui import export_standalone_html, serve_dashboard

            if args.export_only:
                out_path = export_standalone_html(args.suite_dir, args.output)
                print(f"Exported standalone interactive dashboard to {out_path}")
            else:
                serve_dashboard(args.suite_dir, host=args.host, port=args.port, backend=args.backend, show=args.show)
            return 0
        if args.command == "transfer":
            from .transfer import transfer

            transfer(
                args.train_input, args.train_source, args.train_turbine,
                args.target_input, args.target_source, args.target_turbine,
                args.output, args.min_bin_count,
            )
            print(f"verified_real transfer written to {args.output}")
            return 0
        if args.command == "demo":
            with tempfile.TemporaryDirectory() as directory:
                fixture = Path(directory) / "synthetic.csv"
                write_fixture(fixture)
                evidence = run(fixture, args.output, "synthetic", args.turbine, args.min_bin_count)
        else:
            evidence = run(args.input, args.output, args.source, args.turbine, args.min_bin_count)
    except (OSError, ValueError) as error:
        parser.exit(2, f"wintellisyt: {error}\n")
    print(f"{evidence} run written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
