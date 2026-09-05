#!/bin/python3
import os
import subprocess
import argparse
from pathlib import Path


def simulate_csc(csc_file: str, cooja_base: str, output_dir: str | None = None) -> None:
    csc_file = Path(csc_file).resolve()

    original_dir = os.getcwd()
    if output_dir is None:
        output_dir = original_dir
    else:
        output_dir = Path(output_dir).resolve()

    os.chdir(cooja_base)
    subprocess.run(
        [
            "./gradlew",
            "--no-daemon",  # keep the build's JVM in our process group so a
                            # kill of the parent script actually terminates it
            "run",
            f"--args=--no-gui {csc_file} --logdir={output_dir}",
        ],
        check=True,
    )
    os.chdir(original_dir)

    print(f"Finish simulating. Wrote to {output_dir}")


if __name__ == "__main__":
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
    simulate_csc(csc_file=args.csc, cooja_base=args.cooja_base, output_dir=args.output_dir)
