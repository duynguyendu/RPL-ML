#!/bin/python3
import os
import subprocess
import argparse
from pathlib import Path


def simulate(
    csc_file: str, cooja_base: str | None = None, output_dir: str | None = None
):
    original_dir = os.getcwd()
    if output_dir is None:
        output_dir = original_dir

    os.chdir(cooja_base)
    subprocess.run(
        [
            "./gradlew",
            "run",
            f"--args=--no-gui {csc_file} --logdir={output_dir}",
        ],
        check=True,
    )
    os.chdir(original_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run simulation in COOJA")
    ap.add_argument("csc", type=str)
    ap.add_argument(
        "--cooja-base",
        type=str,
        default=None,
        required=True,
        help="The directory of COOJA",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="The directory for output file",
    )
    args = ap.parse_args()

    cooja_base = args.cooja_base
    csc_file = Path(args.csc).resolve()
    output_dir = args.output_dir
    simulate(csc_file, cooja_base, output_dir)

    print(f"Finish simulating. Wrote to {output_dir}")


if __name__ == "__main__":
    main()
