from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def _validate_inputs():
    from biabak import config as C
    required = [("BIABAK_BOREHOLE_XLSX", C.BOREHOLE_XLSX), ("BIABAK_COVARIATE_XLSX", C.COVARIATE_XLSX), ("BIABAK_RASTER_DIR", C.RASTER_DIR), ("BIABAK_GEOLOGY_DIR", C.GEOLOGY_DIR), ("BIABAK_SOIL_SHP", C.SOIL_SHP)]
    missing = [(name, path) for name, path in required if not Path(path).exists()]
    if missing:
        print("BIABAK cannot start because required input data are missing.", file=sys.stderr)
        print("Configure the input paths using local_env.bat before running the workflow.", file=sys.stderr)
        print("See local_env.example.bat and README.md for instructions.", file=sys.stderr)
        for name, path in missing:
            print(f"  {name}: {path}", file=sys.stderr)
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(
        prog="biabak",
        description="BIABAK Transferability Framework"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the complete BIABAK transferability workflow"
    )
    run_parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the computationally intensive benchmark simulations"
    )

    args = parser.parse_args()

    if args.command == "run":
        _validate_inputs()
        command = [sys.executable, str(ROOT / "run_all.py")]

        if args.skip_benchmark:
            command.append("--skip-benchmark")

        raise SystemExit(
            subprocess.call(command, cwd=ROOT)
        )


if __name__ == "__main__":
    main()