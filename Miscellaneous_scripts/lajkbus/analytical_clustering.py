import osmnx as ox
import networkx as nx
import numpy as np

import matplotlib as mp

from typing import Callable, Any


def community_centrality_node_removal(
        graph: nx.MultiGraph,
        inactive_nodes: list[int] | None =None,
        partition: list[set] | None = None,
        scoring: dict | None = None,
        preserve_original_partition: bool = False,
        partitioning_algorithm: Callable[[nx.MultiGraph, ...], list]
        = nx.community.edge_current_flow_betweenness_partition,
        scoring_algorithm: Callable[[nx.MultiGraph], dict]
        = nx.betweenness_centrality,
        partition_size_sensitivity: float = 1,
        **kwargs
) -> tuple[int, dict[int, float], dict[str, dict[Any, Any] | list[set[Any]]]]:
    """
    Function designed to evaluate nodes which are to be removed from stop list.
    Nodes within bigger clusters are more likely to be removed
    while smaller clusters are more likely to be preserved
    :param graph: networkx Multigraph object - a city graph
    :param inactive_nodes: list of stops that were removed so far
    :param partition: if one wants to preserve the original partition
    :param scoring: if one wants to preserve the original scoring
    :param preserve_original_partition: if one wants to preserve the original partition.
    If marked 'False', automatically 'partition' and 'scoring' are set to 'None'
    :param partitioning_algorithm: a partition algorithm. Optional arguments should
    ba passed in kwargs. For the default 'number_of_sets=...'
    :param scoring_algorithm: an algorithm for the base ranking of nodes
    :param partition_size_sensitivity: to balance how impactful is partition size
    to the final score
    :return: tuple: 1) node to be removed; 2) scoring of all remaining nodes
    3) additional data if one wants to use the existing partition
    """
    graph_copy = graph.copy()

    if not preserve_original_partition:
        # recalculate scores and partition
        scoring = None
        partition = None
        # add connections between stops on sides of a removed stop
        for node in inactive_nodes:
            neighbours = list(graph.neighbors(node))
            if len(neighbours) >= 2:
                for j in range(len(neighbours)-1):
                    for i in range(len(neighbours)-1):
                        graph.add_edge(neighbours[j], neighbours[i+1])
        # remove stops which were to be removed
        for node in inactive_nodes:
            graph_copy.remove_node(node)

    # relabel nodes for consecutive numbers as sometimes there are bugs
    relabel_nodes_map = {}
    for num, node in enumerate(graph_copy.nodes):
        relabel_nodes_map[node] = num
    graph_copy = nx.relabel_nodes(graph_copy, relabel_nodes_map)

    if (inactive_nodes is not None) & preserve_original_partition:
        inactive_nodes = [relabel_nodes_map[t] for t in inactive_nodes]

    # compute the evaluation score
    if scoring is None:
        scoring = scoring_algorithm(graph_copy)
    else:
        scoring = {relabel_nodes_map[k]: v for k, v in scoring.items()}

    # compute the partition if not the first iteration
    if partition is None:
        partition = partitioning_algorithm(graph_copy, **kwargs)
    else:
        partition = [{relabel_nodes_map[t] for t in part_set} for part_set in partition]

    # do not consider nodes already removed
    partition = [{i for i in part_set if i not in inactive_nodes} for part_set in partition]

    # create the evaluation metric
    final_scores = {}
    for part_set in partition:
        part_size = len(part_set)
        for node in part_set:
            final_scores[node] = scoring[node] / np.power(part_size, partition_size_sensitivity)

    # relabel back everything
    inverse_relabel_mapping = {v: k for k, v in relabel_nodes_map.items()}
    final_scores = {inverse_relabel_mapping[k]: v for k, v in final_scores.items()}
    auxiliary_data = {
        'scoring': {inverse_relabel_mapping[k]: v for k, v in scoring.items()},
        'partition': [{inverse_relabel_mapping[t] for t in part_set} for part_set in partition]
    }

    # finding the element for the removal
    minimal_value = min(final_scores.values())
    minimal_element = [k for k, v in final_scores.items() if v == minimal_value][0]

    return minimal_element, final_scores, auxiliary_data


# Only for the drawing purposes I need the relabelling here
city_graph = ox.load_graphml('graph_skotniki.graphml')
relabel_nodes_mapping = {}
for num, node in enumerate(city_graph.nodes):
    relabel_nodes_mapping[node] = num
city_graph = nx.relabel_nodes(city_graph, relabel_nodes_mapping)

colour_palette = list(mp.colors.BASE_COLORS.keys())

graph_data = {
    'scoring': None,
    'partition': None
}
removed_stops = []

for step in range(32):
    node_to_remove, node_scores, graph_data = community_centrality_node_removal(
        graph=city_graph,
        inactive_nodes=removed_stops,
        partition=graph_data['partition'],
        scoring=graph_data['scoring'],
        number_of_sets=5
    )

    removed_stops += [node_to_remove]

    if step % 5 == 0:
        colour_list = np.zeros(len(city_graph.nodes))
        for _num, part in enumerate(graph_data['partition']):
            colour_list[list(part)] = _num
        colour_list = [colour_palette[int(t)] for t in colour_list]
        colour_list = ['white' if num in removed_stops else t for num, t in enumerate(colour_list)]
        node_sizes = [50 if num in removed_stops else 10 for num, t in enumerate(colour_list)]

        ox.plot_graph(city_graph, node_color=colour_list, node_size=node_sizes)

