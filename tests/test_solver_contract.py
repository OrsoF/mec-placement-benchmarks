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


def get_base_name(base):
    if isinstance(base, ast.Name):
        return base.id

    if isinstance(base, ast.Attribute):
        return base.attr

    return None


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
        if solver_file.name != "__init__.py"
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
        }, f"{solver_file.name} Solver must inherit GenericSolver or RLSolver"
