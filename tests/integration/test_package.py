"""Imports, CLI execution, and benchmark validation after the src migration."""

import ast
import csv
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mecbench.benchmarks import runner
from mecbench.cli import main
from mecbench.domain.validation import SolutionValidationError
from mecbench.instances.synthetic import create_tree_problem_instance

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
MODULES = sorted(
    ".".join(path.relative_to(SOURCE).with_suffix("").parts).removesuffix(".__init__")
    for path in (SOURCE / "mecbench").rglob("*.py")
)


@pytest.mark.parametrize("module", MODULES)
def test_public_modules_import(module):
    if module.startswith("mecbench.solvers.rl."):
        pytest.importorskip("stable_baselines3")
    if module.startswith(("mecbench.solvers.exact.", "mecbench.solvers.relaxations.")) or module in {
        "mecbench.solvers.arc_formulation", "mecbench.solvers.path_formulation"
    }:
        pytest.importorskip("pyomo")
    if module == "mecbench.solvers.evolutionary.cma":
        pytest.importorskip("cma")
    if module in {"mecbench.environments.spp_env", "mecbench.environments.hrl_env"}:
        pytest.importorskip("gymnasium")
    importlib.import_module(module)


def test_local_imports_and_dynamic_registry_paths_exist():
    for path in (SOURCE / "mecbench").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("mecbench.") and " " not in node.value:
                    modules = [node.value]
            for module in modules:
                if module.startswith("mecbench."):
                    assert importlib.util.find_spec(module) is not None, (path, module)
                else:
                    assert module.split(".")[0] not in {
                        "core", "instances", "solvers", "rl", "data_io", "plotting"
                    }, (path, module)


def test_solver_aliases_and_package_directory():
    from mecbench.benchmarks.io import SOLVER_FILE_ALIASES, SOLVER_MODULES, SOLVER_PATH

    assert Path(SOLVER_PATH).is_dir()
    assert set(SOLVER_FILE_ALIASES.values()) <= set(SOLVER_MODULES)


def run_cli(tmp_path, *args):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE) + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "mecbench", *args], cwd=tmp_path,
        env=environment, capture_output=True, text=True, timeout=60,
    )


def test_cli_help_outside_repository(tmp_path):
    result = run_cli(tmp_path, "--help")
    assert result.returncode == 0, result.stderr
    assert "--instance" in result.stdout
    assert "--solvers" in result.stdout


@pytest.mark.parametrize("instance,expected_users", [("tree", "2"), ("star", "4")])
def test_cli_creates_benchmark_results_outside_repository(tmp_path, instance, expected_users):
    result = run_cli(
        tmp_path, "--instance", instance, "--groups", "2", "--solvers", "fast_heuristic",
        "--start-users", "2", "--stop-users", "3", "--neighbors", "0", "--no-plots",
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    with (tmp_path / "artifacts/data" / f"exp_{instance}.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["users"] == expected_users
    assert rows[0]["solver"] == "fast_heuristic"
    assert float(rows[0]["score"]) >= 0


def test_cli_rejects_invalid_step_before_creating_output(tmp_path):
    result = run_cli(tmp_path, "--step-users", "0")
    assert result.returncode == 2
    assert not (tmp_path / "artifacts").exists()


def test_benchmark_runner_rejects_infeasible_solver_output(monkeypatch):
    problem = create_tree_problem_instance(2, 0)
    candidate = SimpleNamespace(
        run=lambda: None, x_var=problem.x_variable, y_var=problem.y_variable,
        q_var=[flow + 1000 for flow in problem.max_flows],
    )
    monkeypatch.setattr(runner, "make_solver", lambda *args: candidate)
    with pytest.raises(SolutionValidationError):
        runner.run_solver("heuristic", problem, 1)


def test_failed_benchmark_returns_nonzero_status_and_records_failure(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("simulated solver failure")

    monkeypatch.setattr(runner, "run_solver", fail)
    output = tmp_path / "failed.csv"
    assert main([
        "--solvers", "fast_heuristic", "--start-users", "2", "--stop-users", "3",
        "--neighbors", "0", "--no-plots", "--output", str(output),
    ]) == 1
    with output.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["score"] == ""
