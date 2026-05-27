from core.problem_classes.probleminstance import ProblemInstance


def score_state(problem_instance: ProblemInstance, state: list) -> float:
    """Score a state by its served demand quantities."""
    demand_count = problem_instance.total_number_of_demands
    return float(sum(state[-demand_count:]))

