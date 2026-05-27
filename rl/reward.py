from core.problem_classes.problem_instance import ProblemInstance
from rl.state_codec import state_to_vectors


def served_quantity_score(problem_instance: ProblemInstance, state) -> float:
    """Score a state by its served demand quantities."""
    return float(sum(state_to_vectors(problem_instance, state)[2]))


def score_state(problem_instance: ProblemInstance, state) -> float:
    """Backward-compatible alias for served_quantity_score."""
    return served_quantity_score(problem_instance, state)
