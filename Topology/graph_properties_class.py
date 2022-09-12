import networkx as nx
import numpy as np
import warnings
import statistics


def centrality_degree(graph, tuned=True, alpha=1):
    """
    Refined function given in a networkx package allowing to calculate centrality degree on weighted networks
    @param graph: networkx graph
    @param tuned: choose whether to use refined version proposed in
    "Node centrality in weighted networks: Generalizing degree and shortest paths" by Tore Opsahla, Filip Agneessensb,
    John Skvoretzc or more straightforward approach (tuned=True => article version)
    @param alpha: parameter used in refined version of the tuned model
    @return: dictionary with centrality degree per node
    """
    if len(graph) <= 1:
        return {n: 1 for n in graph}
    if not nx.is_weighted(graph):
        return nx.degree_centrality(graph)

    elif nx.is_weighted(graph) and not tuned:
        max_degree_corr = 1 / (max([i[1] for i in graph.degree(weight='weight')]) * (len(graph) - 1))
        return {n: max_degree_corr * d for n, d in graph.degree(weight='weight')}

    elif nx.is_weighted(graph) and tuned:
        degrees_strength = zip(graph.degree(), graph.degree(weight='weight'))
        degrees_strength = [(x[0][0], x[0][1], x[1][1]) for x in degrees_strength]

        def foo(s, k, alpha):
            if k == 0:
                return 0
            else:
                return k * np.power(s / k, alpha)

        return {n: foo(s, k, alpha) for (n, k, s) in degrees_strength}

    else:
        raise Exception("Invalid arguments")


def nodes_neighbours(g, deg=2):
    assert isinstance(g, nx.classes.graph.Graph), "Incorrect type"
    if deg == 1:
        return nx.degree(g)
    elif deg == 2:
        def foo(g, i):
            return set(n for n in g.neighbors(i))

        def foo2(set_list):
            if len(set_list) == 0:
                return set()
            else:
                return set.union(*set_list)

        return {j: foo2([foo(g, t) for t in foo(g, j)]) for j in g.nodes}
    else:
        warnings.warn("Currently not implemented degree, returning None instead")
        return None


def local_rank(g):
    second_neighbours = nodes_neighbours(g, 2)
    r = {key: len(val) for key, val in second_neighbours.items()}
    q = {node: sum([r[t] for t in g.neighbors(node)]) for node in g.nodes}
    return {node: sum([q[t] for t in g.neighbors(node)]) for node in g.nodes}


def h_index(g):
    sorted_neighbor_degrees = {n: sorted((g.degree(v) for v in g.neighbors(n)), reverse=True) for n in g.nodes}
    h = dict.fromkeys(g.nodes)
    for n in g.nodes:
        for i in range(1, len(sorted_neighbor_degrees[n])+1):
            if sorted_neighbor_degrees[n][i-1] < i:
                break
            h[n] = i
    return h


class NetworkStructuralProperties:
    """
    Aggregated functions designed to calculate structural properties of the networks
    """

    def __init__(self, graph, name=""):
        # Initial
        self.G = graph
        self.name = name
        # Explicit centrality measures
        self.centrality_degree = None
        self.eigenvector_centrality = None
        self.local_rank = None
        self.coreness = None
        self.h_index = None
        # Path based centrality measures
        self.closeness_centrality = None
        self.node_efficiency = None
        self.katz_centrality = None
        self.betweenness_centrality = None
        self.current_flow_betweenness_centrality = None

    def __repr__(self):
        return "NetworkStructuralProperties (%r) of the graph with nodes = %r & edges = %r" % \
               (self.name, self.G.number_of_nodes, self.G.number_of_edges)

    def explicit_centrality_measures(self, tuned_degree_centrality=True, alpha_degree_centrality=1):
        self.centrality_degree = centrality_degree(self.G, tuned_degree_centrality, alpha_degree_centrality)
        if nx.is_weighted(self.G):
            self.eigenvector_centrality = nx.eigenvector_centrality_numpy(self.G, weight='weight')
        else:
            self.eigenvector_centrality = nx.eigenvector_centrality_numpy(self.G)
        self.local_rank = local_rank(self.G)
        self.coreness = nx.core_number(self.G)
        self.h_index = h_index(self.G)

    def path_centrality_measures(self):
        self.closeness_centrality = nx.closeness_centrality(self.G)
        self.node_efficiency = {n: statistics.fmean([nx.efficiency(self.G, n, t)
                                                     for t in self.G.nodes if t != n]) for n in self.G.nodes}
        self.katz_centrality = nx.katz_centrality_numpy(self.G)
        if nx.is_weighted(self.G):
            self.betweenness_centrality = nx.betweenness_centrality(self.G, weight='weight')
        else:
            self.betweenness_centrality = nx.betweenness_centrality(self.G)
        if nx.is_weighted(self.G):
            self.current_flow_betweenness_centrality = nx.current_flow_betweenness_centrality(self.G, weight='weight')
        else:
            self.current_flow_betweenness_centrality = nx.current_flow_betweenness_centrality(self.G)


