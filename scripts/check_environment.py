"""Check the local development environment without external packages."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REPOSITORY = "pshreesh7/project-2"
EXPECTED_PYTHON = (3, 11)


def report(label: str, passed: bool, detail: str) -> bool:
    """Print one check and return its result."""
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return passed


def run_git(*arguments: str) -> tuple[bool, str]:
    """Run Git from the project root and capture its output."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)

    output = result.stdout.strip() or result.stderr.strip()
    return result.returncode == 0, output


def repository_matches(remote: str) -> bool:
    """Accept the expected repository's standard HTTPS or SSH URL."""
    normalized = remote.strip().lower().rstrip("/")

    if normalized.endswith(".git"):
        normalized = normalized[:-4]

    allowed = {
        f"https://github.com/{EXPECTED_REPOSITORY}",
        f"git@github.com:{EXPECTED_REPOSITORY}",
        f"ssh://git@github.com/{EXPECTED_REPOSITORY}",
    }
    return normalized in allowed


def main() -> int:
    print("Industrial Defect Detection — Environment Check")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Interpreter: {sys.executable}")
    print()

    checks: list[bool] = []

    checks.append(
        report(
            "Python version",
            sys.version_info[:2] == EXPECTED_PYTHON,
            f"{sys.version.split()[0]} (project target: 3.11)",
        )
    )

    checks.append(
        report(
            "Virtual environment",
            sys.prefix != sys.base_prefix,
            sys.prefix,
        )
    )

    git_available = shutil.which("git") is not None
    checks.append(
        report("Git available", git_available, "git found" if git_available else "git missing")
    )

    if git_available:
        repository_ok, repository_root = run_git("rev-parse", "--show-toplevel")
        exact_root = (
            repository_ok
            and Path(repository_root).resolve() == PROJECT_ROOT
        )
        checks.append(
            report("Repository root", exact_root, repository_root)
        )

        remote_ok, remote = run_git("remote", "get-url", "origin")
        checks.append(
            report(
                "Origin repository",
                remote_ok and repository_matches(remote),
                "expected repository"
                if remote_ok and repository_matches(remote)
                else "origin is missing or does not match PShreesh7/Project-2",
            )
        )

        branch_ok, branch = run_git("symbolic-ref", "--short", "HEAD")
        personal_branch = (
            branch_ok and branch not in {"main", "master"}
        )
        checks.append(
            report("Development branch", personal_branch, branch)
        )

        ignore_ok, ignored_path = run_git(
            "check-ignore", ".venv/environment-check.tmp"
        )
        checks.append(
            report(
                "Virtual environment excluded from Git",
                ignore_ok,
                ignored_path or "add .venv/ to .gitignore",
            )
        )

    print()
    if all(checks):
        print("Environment checks passed.")
        print("ML packages, dataset, and model execution are not checked yet.")
        return 0

    print("Resolve the failed checks before committing this setup.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())