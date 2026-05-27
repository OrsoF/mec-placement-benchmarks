from core.problem_classes.probleminstance import ProblemInstance
from rl.state_codec import state_to_vectors


def score_state(problem_instance: ProblemInstance, state) -> float:
    """Score a state by its served demand quantities."""
    return float(sum(state_to_vectors(problem_instance, state)[2]))
