import argparse
import itertools
import json
from collections import defaultdict

import networkx as nx


class pipeline_runner:
    def __init__(self, args):
        self.args = args

        # directed graph from graph_generation.
        # each edge weight is the index of the target API's input parameter that
        # the source API's output feeds (i.e. sentence_bert's `in_param`), not a score.
        self.graph = nx.read_edgelist(
            args.graph_path,
            create_using=nx.MultiDiGraph(),
            data=(('weight', float),),
        )

    # ------------------------------------------------------------------ #
    # Subgraph building
    # ------------------------------------------------------------------ #
    @staticmethod
    def is_connected_subgraph(subgraph):
        return nx.is_weakly_connected(subgraph)

    @staticmethod
    def remove_duplicate_edges(subgraph):
        """Keep at most one source per (target API, input-parameter index).

        The edge weight is the target's input-parameter index, so deduping by
        (target, weight) ensures a single provider feeds each input parameter.
        """
        unique_edges = set()
        edges_to_remove = []

        for u, v, key, data in subgraph.edges(data=True, keys=True):
            param_idx = data.get('weight', 0)
            edge_id = (v, param_idx)  # (target API, target's input-parameter index)
            if edge_id in unique_edges:
                edges_to_remove.append((u, v, key))
            else:
                unique_edges.add(edge_id)

        for u, v, key in edges_to_remove:
            subgraph.remove_edge(u, v, key)

        return subgraph

    @staticmethod
    def remove_cycles(subgraph):
        """Remove cycles from the subgraph."""
        try:
            while True:
                cycle = next(nx.simple_cycles(subgraph))
                subgraph.remove_edge(cycle[0], cycle[1])
        except StopIteration:
            pass  # No more cycles
        return subgraph

    def generate_subgraphs_with_constraints(self, graph, n):
        """Generate connected, acyclic subgraphs of size n."""
        subgraphs = []
        for nodes in itertools.combinations(graph.nodes(), n):
            subgraph = graph.subgraph(nodes).copy()

            subgraph = self.remove_duplicate_edges(subgraph)
            subgraph = self.remove_cycles(subgraph)
            if not self.is_connected_subgraph(subgraph):
                continue  # Discard subgraph if not connected

            subgraphs.append(subgraph)
        return subgraphs

    def build_subgraphs(self):
        all_graphs = []
        components = list(nx.weakly_connected_components(self.graph))

        for component in components:
            component_subgraph = self.graph.subgraph(component).copy()

            for n in range(self.args.min_size, self.args.max_size + 1):
                if len(component) < n:
                    continue

                subgraphs = self.generate_subgraphs_with_constraints(component_subgraph, n)
                for subgraph in subgraphs:
                    # "weight" here is the target's input-parameter index (see __init__)
                    edge_list = [
                        {"source": u, "target": v, "weight": int(data['weight'])}
                        for u, v, data in subgraph.edges(data=True)
                    ]
                    all_graphs.append(edge_list)

        return all_graphs

    # ------------------------------------------------------------------ #
    # Topological sequences
    # ------------------------------------------------------------------ #
    @staticmethod
    def all_topological_sorts(graph):
        """Compute all possible topological sorts of a directed graph given as an edge list."""
        in_degree = defaultdict(int)
        all_nodes = set()

        for edge in graph:
            u = edge["source"]
            v = edge["target"]
            all_nodes.add(u)
            all_nodes.add(v)
            in_degree[v] += 1
            if u not in in_degree:
                in_degree[u] = 0  # Ensure all nodes are in in_degree

        result = []

        def backtrack(path):
            if len(path) == len(all_nodes):
                result.append(list(path))
                return

            for node in all_nodes:
                if in_degree[node] == 0 and node not in path:
                    path.append(node)
                    for edge in graph:
                        if edge["source"] == node:
                            in_degree[edge["target"]] -= 1

                    backtrack(path)

                    path.pop()
                    for edge in graph:
                        if edge["source"] == node:
                            in_degree[edge["target"]] += 1

        backtrack([])
        return result

    def build_sequences(self, subgraphs):
        all_topo = []
        for g in subgraphs:
            result = self.all_topological_sorts(g)
            if result:
                all_topo.append({"subgraph": g, "topo_sort": result})
        return all_topo

    # ------------------------------------------------------------------ #
    def run(self):
        subgraphs = self.build_subgraphs()
        sequences = self.build_sequences(subgraphs)

        with open(self.args.output_path, 'w') as f:
            json.dump(sequences, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', type=str, required=True, help='input graph edgelist (weighted, directed)')
    parser.add_argument('--output_path', type=str, required=True, help='output json: list of {subgraph, topo_sort}')
    parser.add_argument('--min_size', type=int, default=2, help='minimum subgraph size (number of nodes)')
    parser.add_argument('--max_size', type=int, default=4, help='maximum subgraph size (number of nodes)')

    args = parser.parse_args()

    runner = pipeline_runner(args)
    runner.run()
