from numbers import Integral, Real


class Demand:
    """
    A demand is a certain amount of resources needed at a destination
    node and a flow in [flow_min, flow_max] that need to transit.
    """

    def __init__(
        self,
        origin_node_index: int,
        flow_min: float,
        flow_max: float,
        max_latency: float,
    ):
        if not isinstance(origin_node_index, Integral) or isinstance(
            origin_node_index, bool
        ):
            raise TypeError("origin_node_index must be an int.")
        if not isinstance(flow_min, Real):
            raise TypeError("flow_min must be numeric.")
        if not isinstance(flow_max, Real):
            raise TypeError("flow_max must be numeric.")
        if not isinstance(max_latency, Real):
            raise TypeError("max_latency must be numeric.")
        if origin_node_index < 0:
            raise ValueError("origin_node_index must be non-negative.")
        if flow_min < 0:
            raise ValueError("flow_min must be non-negative.")
        if flow_max < 0:
            raise ValueError("flow_max must be non-negative.")
        if flow_min > flow_max:
            raise ValueError("flow_min must be <= flow_max.")
        if max_latency < 0:
            raise ValueError("max_latency must be non-negative.")

        self.origin_node_index = origin_node_index
        self.flow_min = flow_min
        self.flow_max = flow_max
        self.max_latency = max_latency
