import numpy as np

from mecbench.domain.conversions import split_state


def served_quantity(problem, s) -> float:
    return float(np.sum(split_state(problem, s)[2]))


def placement_routing_score(problem, checker, rng, s, steps: int, increment: int) -> float:
    p, r, q = split_state(problem, s)
    p = p.astype(int).tolist()
    r = r.astype(int).tolist()
    q = list(problem.min_flows)
    if not checker.test_solution(p, r, q):
        return -1.0

    D = problem.total_number_of_demands
    for _ in range(steps):
        d = int(rng.integers(D))
        if q[d] >= problem.max_flows[d]:
            continue
        dq = min(increment, problem.max_flows[d] - q[d])
        q[d] += dq
        if not checker.test_solution(p, r, q):
            q[d] -= dq
    return float(sum(q))


def normalized_delta(new_score: float, old_score: float) -> float:
    return float((new_score - old_score) / 100.0)
