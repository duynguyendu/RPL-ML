#!/bin/python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path

INIT_SCRIPT = Path(__file__).resolve().parent / "gradle_build_dir_init.gradle"


def _gradle_dirs(cache_root: Path) -> tuple[Path, Path, Path]:
    return cache_root / "home", cache_root / "project-cache", cache_root / "build-output"


def _run_gradle(cooja_base: Path, cache_root: Path, *task_args: str) -> None:
    home, project_cache, build_output = _gradle_dirs(cache_root)
    for d in (home, project_cache, build_output):
        d.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GRADLE_USER_HOME"] = str(home)
    env["COOJA_BUILD_DIR"] = str(build_output)
    subprocess.run(
        [
            "./gradlew",
            "--no-daemon",
            f"--project-cache-dir={project_cache}",
            f"--init-script={INIT_SCRIPT}",
            *task_args,
        ],
        cwd=cooja_base,
        check=True,
        env=env,
    )


def ensure_template_built(cooja_base: str) -> Path:
    """Compile Cooja's own (never-job-specific) Java sources exactly once
    into a shared, read-only template under `.gradle_home/_template`. Every
    concurrent slot seeds its private Gradle state from this instead of each
    independently cold-compiling the same unchanging code -- see
    `seed_slot_from_template`.

    Not lock-protected: NFS file locking is exactly what caused the Gradle
    lock-timeout failures this whole per-slot isolation scheme works around,
    so we don't lean on it here too. Safe as long as this is called once,
    synchronously, before any concurrent jobs are dispatched -- which
    ansible/simulate_seeds.yml does via a `run_once` task (ansible's play
    ordering, not a filesystem lock, is what actually serializes this across
    hosts); simulate.py's own main() also calls this before its dispatch
    loop starts, as a fallback for plain single-machine runs.
    """
    cooja_base = Path(cooja_base).resolve()
    template_dir = cooja_base / ".gradle_home" / "_template"
    marker = template_dir / ".built"
    if marker.exists():
        return template_dir
    print(f"=== Building shared Cooja template (first use) at {template_dir} ===")
    _run_gradle(cooja_base, template_dir, "compileJava")
    marker.touch()
    return template_dir


def seed_slot_from_template(cooja_base: str, build_dir_name: str) -> Path:
    """Give `build_dir_name`'s slot its own private copy of the shared
    template (see `ensure_template_built`) the first time it's used, so its
    first gradlew invocation already has up-to-date compiled classes instead
    of recompiling Cooja's Java sources from scratch. A plain file copy is
    safe for many slots to do concurrently (they all only *read* the
    template, which is never modified after the initial build), unlike
    Gradle's own mutable-state caches.
    """
    cooja_base = Path(cooja_base).resolve()
    slot_cache_dir = cooja_base / ".gradle_home" / build_dir_name
    if not slot_cache_dir.exists():
        template_dir = ensure_template_built(cooja_base)
        shutil.copytree(template_dir, slot_cache_dir)
    return slot_cache_dir


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
    #   3. the actual build OUTPUT directory (<cooja_base>/build/, e.g.
    #      compiled Cooja .class files) -- this is where compileJava writes,
    #      and it's fixed by the project's build.gradle, not by either cache
    #      setting above. With isolated caches (1) and (2) but a still-shared
    #      output dir, every slot thinks it needs to rebuild Cooja's own Java
    #      code (no shared up-to-date state) and they race on the same
    #      output tree ("Unable to delete file '.../build/classes/...'").
    #      Redirected via gradle_build_dir_init.gradle (COOJA_BUILD_DIR env
    #      var), since Gradle has no CLI flag for this.
    # Multiple concurrent gradlew invocations (multiple slots per host, and
    # $HOME may be shared across hosts) fighting over any of these fail with
    # lock timeouts or file races. Give each slot its own copy of all three
    # so they never collide -- seeded from a shared template (see
    # seed_slot_from_template) so that isolation doesn't also mean every slot
    # separately re-compiles Cooja's own (never-changing) Java code.
    env = os.environ.copy()
    extra_args = []
    if build_dir_name:
        slot_cache_dir = seed_slot_from_template(cooja_base, build_dir_name)
        gradle_home, project_cache_dir, build_output_dir = _gradle_dirs(slot_cache_dir)
        env["GRADLE_USER_HOME"] = str(gradle_home)
        env["COOJA_BUILD_DIR"] = str(build_output_dir)
        extra_args.append(f"--project-cache-dir={project_cache_dir}")
        extra_args.append(f"--init-script={INIT_SCRIPT}")

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
