from core.problem_classes.problem_instance import ProblemInstance
from core.problem_classes.solving_instance import SolvingInstance
import numpy as np
import time
from functools import wraps


class GenericSolver:
    REQUIRED_OUTPUT_ATTRIBUTES = (
        "x_var",
        "y_var",
        "q_var",
        "score",
        "runtime",
        "build_time",
    )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        if getattr(original_init, "_solver_contract_wrapped", False):
            return

        @wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._ensure_output_contract()

        wrapped_init._solver_contract_wrapped = True
        cls.__init__ = wrapped_init

    def __init__(self, problem_instance: ProblemInstance, **args) -> None:
        self.problem_instance = problem_instance
        self.solver_name = "generic_solver"

        self.x_var: np.ndarray | None = None
        self.y_var: np.ndarray | None = None
        self.q_var: np.ndarray | None = None

        self.score: float | None = None
        self.runtime: float | None = None
        self.build_time: float = 0

    def _ensure_output_contract(self):
        for attribute in self.REQUIRED_OUTPUT_ATTRIBUTES:
            if hasattr(self, attribute):
                continue

            if attribute == "build_time":
                setattr(self, attribute, 0.0)
            else:
                setattr(self, attribute, None)

    @staticmethod
    def _value(x) -> float:
        value = getattr(x, "value", x)
        if value is None:
            return 0.0
        return float(value)

    def _set_output_variables(self, x_var, y_var, q_var) -> None:
        self.x_var = np.asarray(x_var, dtype=float)
        self.y_var = np.asarray(y_var, dtype=float)
        self.q_var = np.asarray(q_var, dtype=float)

        # Legacy aliases still used by older experiment helpers.
        self.x_variable = self.x_var
        self.y_variable = self.y_var
        self.q_variable = self.q_var

    def _set_problem_baseline_solution(self) -> None:
        self._set_output_variables(
            self.problem_instance.x_variable,
            self.problem_instance.y_variable,
            self.problem_instance.q_variable,
        )
        self.score = float(np.sum(self.q_var))

    def _set_arc_model_solution(
        self,
        x,
        y,
        q,
        demand_count: int,
        arc_count: int,
        service_count: int,
        node_count: int,
    ) -> None:
        self._set_output_variables(
            [
                [self._value(x[d, a]) for a in range(arc_count)]
                for d in range(demand_count)
            ],
            [
                [self._value(y[s, u]) for u in range(node_count)]
                for s in range(service_count)
            ],
            [self._value(q[d]) for d in range(demand_count)],
        )

    def _set_path_model_solution(
        self,
        l,
        y,
        q,
        paths: list[list[int]],
        path_to_demand: list[int],
        arc_index: dict,
        demand_count: int,
        arc_count: int,
        service_count: int,
        node_count: int,
    ) -> None:
        x_var = np.zeros((demand_count, arc_count))
        for p, path in enumerate(paths):
            weight = self._value(l[p])
            d = path_to_demand[p]
            for i in range(len(path) - 1):
                x_var[d, arc_index[(path[i], path[i + 1])]] += weight

        y_var = [
            [self._value(y[s, u]) for u in range(node_count)]
            for s in range(service_count)
        ]
        q_var = [self._value(q[d]) for d in range(demand_count)]
        self._set_output_variables(x_var, y_var, q_var)

    def run(
        self,
        initial_variable: tuple,
    ):
        # To be implemented in the real solver.
        assert False, "To be implemented."

    def _start_solving(self):
        self.start_time = time.time()

    def _end_solving(self):
        assert (
            self.x_var is not None
            and self.y_var is not None
            and self.q_var is not None
        ), "Result variables not set."
        self.runtime = time.time() - self.start_time
