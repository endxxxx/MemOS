"""Bootstrap SkillFlow_Test verifier dependencies before evaluation."""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SKILLFLOW_TEST_ROOT = Path("/root/SkillFlow_Test")

APT_PACKAGES: tuple[str, ...] = (
    "gnumeric",  # ssconvert --recalc for Excel formula verification
    "nodejs",  # test_outputs.js verifiers
)

PIP_PACKAGES: tuple[str, ...] = (
    "pytest",
    "pytest-json-ctrf",
    "openpyxl",
    "pandas",
    "numpy",
    "scipy",
    "python-docx",
)

IMPORT_CHECKS: dict[str, str] = {
    "pytest": "pytest",
    "openpyxl": "openpyxl",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "python-docx": "docx",
}


@dataclass(frozen=True)
class DependencyAction:
    name: str
    kind: str
    installed: bool
    detail: str


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def _module_importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def _run_command(command: list[str], *, description: str) -> None:
    print(f"[skillflow-deps] {description}: {' '.join(command)}")
    subprocess.run(command, check=True)


def _apt_package_installed(package: str) -> bool:
    proc = subprocess.run(
        ["dpkg-query", "-W", "-f=${Status}", package],
        capture_output=True,
        text=True,
        check=False,
    )
    return "install ok installed" in (proc.stdout or "")


def _ensure_apt_packages() -> list[DependencyAction]:
    actions: list[DependencyAction] = []
    missing = [pkg for pkg in APT_PACKAGES if not _apt_package_installed(pkg)]
    for pkg in APT_PACKAGES:
        if pkg not in missing:
            actions.append(
                DependencyAction(name=pkg, kind="apt", installed=True, detail="already installed")
            )

    if not missing:
        return actions

    if not _command_exists("apt-get"):
        raise RuntimeError(
            "Missing apt packages for SkillFlow verifiers and apt-get is unavailable: "
            + ", ".join(missing)
        )

    _run_command(
        ["apt-get", "update", "-qq"],
        description="refresh apt package index",
    )
    _run_command(
        ["apt-get", "install", "-y", "-qq", *missing],
        description="install apt packages",
    )

    for pkg in missing:
        actions.append(
            DependencyAction(
                name=pkg,
                kind="apt",
                installed=_apt_package_installed(pkg),
                detail="installed",
            )
        )
    return actions


def _pip_distribution_ready(package: str) -> bool:
    if package == "pytest-json-ctrf":
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        return "--ctrf" in (proc.stdout or "")

    module_name = IMPORT_CHECKS.get(package)
    if module_name:
        return _module_importable(module_name)

    proc = subprocess.run(
        [sys.executable, "-m", "pip", "show", package],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _missing_pip_packages() -> list[str]:
    return [package for package in PIP_PACKAGES if not _pip_distribution_ready(package)]


def _ensure_pip_packages() -> list[DependencyAction]:
    actions: list[DependencyAction] = []
    missing = _missing_pip_packages()
    for package in PIP_PACKAGES:
        if package not in missing:
            actions.append(
                DependencyAction(name=package, kind="pip", installed=True, detail="already importable")
            )

    if not missing:
        return actions

    _run_command(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *missing],
        description="install python packages",
    )

    still_missing = _missing_pip_packages()
    if still_missing:
        raise RuntimeError(
            "SkillFlow verifier python dependencies are still missing after install: "
            + ", ".join(still_missing)
        )

    for package in missing:
        actions.append(
            DependencyAction(name=package, kind="pip", installed=True, detail="installed")
        )
    return actions


def _node_modules_ready() -> bool:
    package_json = SKILLFLOW_TEST_ROOT / "package.json"
    if not package_json.is_file():
        return True
    return (SKILLFLOW_TEST_ROOT / "node_modules" / "xlsx" / "package.json").is_file()


def _ensure_node_dependencies() -> list[DependencyAction]:
    actions: list[DependencyAction] = []
    package_json = SKILLFLOW_TEST_ROOT / "package.json"
    if not package_json.is_file():
        return actions

    if not _command_exists("node"):
        raise RuntimeError("node is required for SkillFlow JS verifiers but was not found in PATH")
    if not _command_exists("npm"):
        raise RuntimeError("npm is required for SkillFlow JS verifiers but was not found in PATH")

    if _node_modules_ready():
        actions.append(
            DependencyAction(
                name="xlsx",
                kind="npm",
                installed=True,
                detail="node_modules already present",
            )
        )
        return actions

    _run_command(
        ["npm", "install", "--prefix", str(SKILLFLOW_TEST_ROOT), "--no-audit", "--no-fund"],
        description="install SkillFlow_Test node modules",
    )

    if not _node_modules_ready():
        raise RuntimeError(
            f"Failed to install SkillFlow JS verifier deps under {SKILLFLOW_TEST_ROOT}"
        )

    actions.append(
        DependencyAction(name="xlsx", kind="npm", installed=True, detail="installed")
    )
    return actions


def _verify_runtime_commands() -> list[DependencyAction]:
    checks: list[tuple[str, Callable[[], bool], str]] = [
        ("python3", lambda: _command_exists(sys.executable), "python interpreter"),
        ("bash", lambda: _command_exists("bash"), "test.sh runner"),
        ("ssconvert", lambda: _command_exists("ssconvert"), "Excel formula recalculation"),
        ("node", lambda: _command_exists("node"), "JS verifier runtime"),
    ]
    actions: list[DependencyAction] = []
    missing: list[str] = []
    for name, checker, detail in checks:
        ok = checker()
        actions.append(
            DependencyAction(
                name=name,
                kind="command",
                installed=ok,
                detail=detail,
            )
        )
        if not ok:
            missing.append(name)

    if missing:
        raise RuntimeError(
            "SkillFlow verifier runtime commands are still missing: " + ", ".join(missing)
        )
    return actions


def ensure_skillflow_verifier_dependencies(
    *,
    test_root: Path = SKILLFLOW_TEST_ROOT,
) -> dict[str, list[dict[str, str | bool]]]:
    """Install and verify dependencies required by SkillFlow_Test verifiers."""
    if not _env_bool("SKILLFLOW_VERIFIER_DEPS_ENABLED", True):
        print("[skillflow-deps] dependency bootstrap skipped (SKILLFLOW_VERIFIER_DEPS_ENABLED=false)")
        return {"skipped": True, "actions": []}

    if not test_root.is_dir():
        raise FileNotFoundError(f"SkillFlow test root not found: {test_root}")

    print("[skillflow-deps] ensuring SkillFlow verifier dependencies")
    actions: list[DependencyAction] = []
    actions.extend(_ensure_apt_packages())
    actions.extend(_ensure_pip_packages())
    actions.extend(_ensure_node_dependencies())
    actions.extend(_verify_runtime_commands())

    summary = [
        {
            "name": action.name,
            "kind": action.kind,
            "installed": action.installed,
            "detail": action.detail,
        }
        for action in actions
    ]
    print(
        "[skillflow-deps] ready "
        f"(apt={sum(1 for a in actions if a.kind == 'apt')}, "
        f"pip={sum(1 for a in actions if a.kind == 'pip')}, "
        f"npm={sum(1 for a in actions if a.kind == 'npm')})"
    )
    return {"skipped": False, "actions": summary}
