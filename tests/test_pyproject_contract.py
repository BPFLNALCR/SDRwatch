from __future__ import annotations

import re
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"


def _load_pyproject() -> dict:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _dependency_names(entries: list[str]) -> set[str]:
    names: set[str] = set()
    for entry in entries:
        name = re.split(r"[<>=!~\s\[]", entry, maxsplit=1)[0].strip().lower()
        if name:
            names.add(name)
    return names


def test_pyproject_exists_and_uses_setuptools_build_backend() -> None:
    data = _load_pyproject()

    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    assert "setuptools>=68" in data["build-system"]["requires"]


def test_project_metadata_covers_no_hardware_runtime_imports() -> None:
    data = _load_pyproject()
    project = data["project"]
    dependency_names = _dependency_names(project["dependencies"])

    assert project["name"] == "sdrwatch"
    assert project["requires-python"] == ">=3.11"
    assert {"flask", "numpy"}.issubset(dependency_names)


def test_dev_extra_contains_test_and_quality_tools() -> None:
    data = _load_pyproject()
    dev_dependency_names = _dependency_names(data["project"]["optional-dependencies"]["dev"])

    assert {"pytest", "ruff", "mypy"}.issubset(dev_dependency_names)


def test_hardware_and_os_managed_dependencies_stay_out_of_base_runtime() -> None:
    data = _load_pyproject()
    dependency_names = _dependency_names(data["project"]["dependencies"])
    rtl_dependency_names = _dependency_names(data["project"]["optional-dependencies"]["rtl"])

    assert "pyrtlsdr" not in dependency_names
    assert "scipy" not in dependency_names
    assert "soapysdr" not in dependency_names
    assert "pyrtlsdr" in rtl_dependency_names