#!/bin/python3
import os
import subprocess
import argparse
from pathlib import Path


def simulate_csc(
    csc_file: str,
    cooja_base: str,
    output_dir: str | None = None,
    build_dir_name: str | None = None,
) -> None:
    csc_file = Path(csc_file).resolve()

    original_dir = os.getcwd()
    if output_dir is None:
        output_dir = original_dir
    else:
        output_dir = Path(output_dir).resolve()

    # Gradle keeps two separate caches, each with its own lock file, and both
    # are shared resources unrelated to our per-slot BUILD_DIR firmware
    # isolation:
    #   1. the user-home cache (~/.gradle/caches/.../fileHashes.lock) --
    #      controlled by GRADLE_USER_HOME;
    #   2. a *project-local* cache inside cooja_base itself
    #      (<cooja_base>/.gradle/.../executionHistory.lock) -- controlled by
    #      --project-cache-dir, NOT by GRADLE_USER_HOME. Since every job runs
    #      gradlew from this same cooja_base (the shared contiki-ng
    #      checkout), concurrent slots collide here too.
    # Multiple concurrent gradlew invocations (multiple slots per host, and
    # $HOME may be shared across hosts) fighting over either lock fail with
    # "Timeout waiting to lock ... cache". Give each slot its own copy of both
    # so they never collide.
    env = os.environ.copy()
    extra_args = []
    if build_dir_name:
        slot_cache_dir = Path(cooja_base).resolve() / ".gradle_homes" / build_dir_name
        gradle_home = slot_cache_dir / "home"
        project_cache_dir = slot_cache_dir / "project-cache"
        gradle_home.mkdir(parents=True, exist_ok=True)
        project_cache_dir.mkdir(parents=True, exist_ok=True)
        env["GRADLE_USER_HOME"] = str(gradle_home)
        extra_args.append(f"--project-cache-dir={project_cache_dir}")

    os.chdir(cooja_base)
    subprocess.run(
        [
            "./gradlew",
            "--no-daemon",  # keep the build's JVM in our process group so a
                            # kill of the parent script actually terminates it
            *extra_args,
            "run",
            f"--args=--no-gui {csc_file} --logdir={output_dir}",
        ],
        check=True,
        env=env,
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
