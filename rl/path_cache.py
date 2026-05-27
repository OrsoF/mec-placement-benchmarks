import networkx as nx


class ShortestPathCache:
    """Memoize indexed shortest paths for one graph."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self._paths: dict[tuple[int, int, int], list] = {}

    def nth_shortest_path(self, source: int, destination: int, index: int) -> list:
        index = int(index)
        cache_key = (source, destination, index)
        if cache_key in self._paths:
            return self._paths[cache_key]

        for path_index, path in enumerate(
            nx.algorithms.simple_paths.shortest_simple_paths(
                self.graph,
                source,
                destination,
            )
        ):
            if path_index == index:
                self._paths[cache_key] = path
                return path

        raise IndexError(
            f"No shortest path at index {index} from {source} to {destination}."
        )

