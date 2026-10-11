"""Missing extras may skip; installed but broken dependencies must fail."""

from types import SimpleNamespace

import pytest

from tests import conftest


def dependency_item(*, require_extras=False):
    return SimpleNamespace(
        iter_markers=lambda name: [pytest.mark.requires("optional_package").mark],
        config=SimpleNamespace(getoption=lambda name: require_extras),
        nodeid="test_optional",
    )


def test_absent_optional_dependency_is_explicit_skip(monkeypatch):
    monkeypatch.setattr(conftest.importlib.util, "find_spec", lambda module: None)
    with pytest.raises(pytest.skip.Exception, match="optional_package"):
        conftest.pytest_runtest_setup(dependency_item())


def test_required_missing_extra_fails(monkeypatch):
    monkeypatch.setattr(conftest.importlib.util, "find_spec", lambda module: None)
    with pytest.raises(pytest.fail.Exception, match="optional_package"):
        conftest.pytest_runtest_setup(dependency_item(require_extras=True))


def test_installed_broken_dependency_is_not_skipped(monkeypatch):
    monkeypatch.setattr(conftest.importlib.util, "find_spec", lambda module: object())

    def broken_import(module):
        raise ModuleNotFoundError("broken transitive dependency", name="transitive_package")

    monkeypatch.setattr(conftest.importlib, "import_module", broken_import)
    with pytest.raises(ModuleNotFoundError, match="broken transitive dependency"):
        conftest.pytest_runtest_setup(dependency_item())
