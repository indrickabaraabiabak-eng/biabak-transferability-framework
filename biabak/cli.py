from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


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
        command = [sys.executable, str(ROOT / "run_all.py")]

        if args.skip_benchmark:
            command.append("--skip-benchmark")

        raise SystemExit(
            subprocess.call(command, cwd=ROOT)
        )


if __name__ == "__main__":
    main()