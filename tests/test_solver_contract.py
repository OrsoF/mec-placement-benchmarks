import ast
from pathlib import Path

from core.generic_solver import GenericSolver


REQUIRED_SOLVER_ATTRIBUTES = {
    "x_var",
    "y_var",
    "q_var",
    "score",
    "runtime",
    "build_time",
}
NON_SOLVER_MODULES = {"arc_formulation.py", "path_formulation.py"}


def get_base_name(base):
    if isinstance(base, ast.Name):
        return base.id

    if isinstance(base, ast.Attribute):
        return base.attr

    return None


def is_solver_module(path):
    return path.name != "__init__.py" and path.name not in NON_SOLVER_MODULES


def test_generic_solver_declares_required_output_attributes():
    assert set(GenericSolver.REQUIRED_OUTPUT_ATTRIBUTES) == REQUIRED_SOLVER_ATTRIBUTES


def test_solver_subclasses_expose_required_output_attributes_without_super_init():
    class BareSolver(GenericSolver):
        def __init__(self):
            self.solver_name = "bare"

    solver = BareSolver()

    for attribute in REQUIRED_SOLVER_ATTRIBUTES:
        assert hasattr(solver, attribute)

    assert solver.x_var is None
    assert solver.y_var is None
    assert solver.q_var is None
    assert solver.score is None
    assert solver.runtime is None
    assert solver.build_time == 0.0


def test_every_solver_module_uses_contract_base_class():
    solver_dir = Path(__file__).resolve().parents[1] / "solvers"
    solver_files = sorted(
        solver_file
        for solver_file in solver_dir.rglob("*.py")
        if is_solver_module(solver_file)
    )

    assert solver_files

    for solver_file in solver_files:
        tree = ast.parse(solver_file.read_text())
        solver_class = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "Solver"
            ),
            None,
        )

        assert solver_class is not None, f"{solver_file.name} has no Solver class"

        base_names = {get_base_name(base) for base in solver_class.bases}
        assert base_names & {
            "GenericSolver",
            "RLSolver",
            "A2CSolver",
            "PPOSolver",
            "PPOOptionsSolver",
            "HeuristicSolver",
        }, f"{solver_file.name} Solver must inherit a known solver base"


def test_solver_run_methods_populate_output_variables():
    solver_dir = Path(__file__).resolve().parents[1] / "solvers"
    solver_files = sorted(
        solver_file
        for solver_file in solver_dir.rglob("*.py")
        if is_solver_module(solver_file)
    )
    output_helpers = {
        "_set_output_variables",
        "_set_arc_model_solution",
        "_set_path_model_solution",
        "_set_problem_baseline_solution",
        "solution_to_arc_variables",
        "_state_to_variables",
    }

    for solver_file in solver_files:
        tree = ast.parse(solver_file.read_text())
        solver_class = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "Solver"
            ),
            None,
        )
        run_method = next(
            (
                node
                for node in solver_class.body
                if isinstance(node, ast.FunctionDef) and node.name == "run"
            ),
            None,
        )
        if run_method is None:
            continue

        assigned = set()
        helper_calls = set()
        for node in ast.walk(run_method):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    targets = target.elts if isinstance(target, ast.Tuple) else [target]
                    for item in targets:
                        if (
                            isinstance(item, ast.Attribute)
                            and isinstance(item.value, ast.Name)
                            and item.value.id == "self"
                        ):
                            assigned.add(item.attr)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                helper_calls.add(node.func.attr)

        assert (
            {"x_var", "y_var", "q_var"} <= assigned
            or helper_calls & output_helpers
        ), f"{solver_file.name} run() must populate x_var, y_var, and q_var"
