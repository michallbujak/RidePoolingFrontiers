import networkx as nx
import numpy as np
import warnings


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


def nodes_neighbours(G, deg=2):
    assert isinstance(G, nx.classes.graph.Graph), "Incorrect type"
    if deg == 1:
        return nx.degree(G)
    elif deg == 2:
        def foo(g, i):
            return set(n for n in g.neighbors(i))

        def foo2(set_list):
            if len(set_list) == 0:
                return set()
            else:
                return set.union(*set_list)

        return {j: foo2([foo(G, t) for t in foo(G, j)]) for j in G.nodes}
    else:
        warnings.warn("Currently not implemented degree, returning None instead")
        return None


def local_rank(G):
    second_neighbours = nodes_neighbours(G, 2)
    r = {key: len(val) for key, val in second_neighbours.items()}
    q = {node: sum([r[t] for t in G.neighbors(node)]) for node in G.nodes}
    return {node: sum([q[t] for t in G.neighbors(node)]) for node in G.nodes}


class StructuralProperties:
    """
    Aggregated functions designed to calculate structural properties of the networks
    """

    def __init__(self, graph):
        self.G = graph
        self.centrality_degree = None
        self.eigenvector_centrality = None

    def __repr__(self):
        return "StructuralProperties of the graph with nodes = %r & edges = %r" % \
               (self.G.number_of_nodes, self.G.number_of_edges)

    def centrality_measures(self, tuned_degree_centrality=True, alpha_degree_centrality=1):
        self.centrality_degree = centrality_degree(self.G, tuned_degree_centrality, alpha_degree_centrality)
        if nx.is_weighted(self.G):
            self.eigenvector_centrality = nx.eigenvector_centrality_numpy(self.G, weight='weight')
        else:
            self.eigenvector_centrality = nx.eigenvector_centrality_numpy(self.G)
