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
        self.origin_node_index = origin_node_index
        self.flow_min = flow_min
        self.flow_max = flow_max
        self.max_latency = max_latency
