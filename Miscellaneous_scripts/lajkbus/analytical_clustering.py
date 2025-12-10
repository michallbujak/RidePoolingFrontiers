import osmnx as ox
import networkx as nx
import numpy as np

import matplotlib as mp

from typing import Callable, Any

import pandas as pd


def community_centrality_node_removal(
        graph: nx.MultiGraph,
        inactive_nodes: list[int] | None =None,
        partition: list[set] | None = None,
        scoring: dict | None = None,
        preserve_original_partition: bool = True,
        preserve_original_scoring: bool = False,
        partitioning_algorithm: Callable[[nx.MultiGraph, ...], list]
        = nx.community.edge_current_flow_betweenness_partition,
        scoring_algorithm: Callable[[nx.MultiGraph], dict]
        = nx.closeness_centrality,
        scoring_within_partitions: bool = False,
        boost_neighbours: bool = True,
        boost_neighbours_factor: float = 0.7,
        initial_scores: dict = {},
        partition_size_sensitivity: float = 0.05,
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
    If marked 'False', automatically 'partition' is set to 'None'
    :param preserve_original_scoring: preserve the scoring calculated in the first run.
    :param partitioning_algorithm: a partition algorithm. Optional arguments should
    ba passed in kwargs. For the default 'number_of_sets=...'
    :param scoring_algorithm: an algorithm for the base ranking of nodes
    :param scoring_within_partitions: calculate score of each node within cluster
    :param boost_neighbours: if True, neighbours of a removed node have their score boosted
    :param boost_neighbours_factor: if boost_neighbours is True, select a fraction of original score
    that is transferred to the neighbouring nodes
    :param initial_scores: required for neighbours boosting
    :param partition_size_sensitivity: to balance how impactful is partition size
    to the final score
    :return: tuple: 1) node to be removed; 2) scoring of all remaining nodes
    3) additional data if one wants to use the existing partition
    """
    def boost_neighbours_func(_graph, _node, _current_scores, _initial_scores, _boost_factor):
        first_order_neighbours = list(_graph.neighbors(_node))
        for first_neighbour in first_order_neighbours:
            _current_scores[first_neighbour] += _boost_factor * _initial_scores[_node]
            if _boost_factor < 1:
                second_order_neighbours = list(_graph.neighbors(first_neighbour))
                for second_neighbour in second_order_neighbours:
                    _current_scores[second_neighbour] += _boost_factor * _boost_factor * _initial_scores[_node]

        return _current_scores

    graph_current = graph.copy()

    if not preserve_original_partition:
        # recalculate partition
        partition = None

    if not preserve_original_scoring:
        # recalculate node scores
        scoring = None

    # add connections between stops on sides of a removed nodes if one wants to recalculate properties
    if (partition is None) | (scoring is None):
        for node in inactive_nodes:
            neighbours = list(graph_current.neighbors(node))
            neighbours = [t for t in neighbours if t not in inactive_nodes]
            if len(neighbours) >= 2:
                for j in range(len(neighbours)-1):
                    for i in range(len(neighbours)-1):
                        graph_current.add_edge(neighbours[j], neighbours[i+1])

    original_graph = graph_current.copy()
    # remove nodes which were to be removed
    for node in inactive_nodes:
        graph_current.remove_node(node)

    # relabel nodes for consecutive numbers as sometimes there are bugs
    relabel_nodes_map = {}
    for num, node in enumerate(graph_current.nodes):
        relabel_nodes_map[node] = num
    graph_current = nx.relabel_nodes(graph_current, relabel_nodes_map)
    inverse_relabel_mapping = {v: k for k, v in relabel_nodes_map.items()}

    # compute the partition if not the first iteration
    if partition is None:
        partition = partitioning_algorithm(graph_current, **kwargs)

    # compute the evaluation score
    if scoring is None:
        scoring = {_node: 0 for _node in original_graph.nodes}
        if scoring_within_partitions:
            scoring_current = {}
            for cluster in partition:
                subgraph = graph_current.subgraph(cluster)
                scoring_current.update(scoring_algorithm(subgraph))
        else:
            scoring_current = scoring_algorithm(graph_current)

        scoring.update({inverse_relabel_mapping[k]: v for k, v in scoring_current.items()})

        if not initial_scores:
            initial_scores = scoring.copy()

        if boost_neighbours:
            for node in inactive_nodes:
                scoring = boost_neighbours_func(
                    _graph=original_graph,
                    _node=node,
                    _current_scores=scoring,
                    _initial_scores=initial_scores,
                    _boost_factor=boost_neighbours_factor)

    # do not consider nodes already removed
    partition_new = [{i for i in part_set if i not in inactive_nodes} for part_set in partition]

    # create the evaluation metric
    final_scores = {}
    for part_set in partition_new:
        part_size = len(part_set)
        for node in part_set:
            final_scores[node] = scoring[node] / np.power(part_size, partition_size_sensitivity)

    # finding the element for the removal
    minimal_value = min(final_scores.values())
    minimal_element = [k for k, v in final_scores.items() if v == minimal_value][0]

    # save scoring and partition
    auxiliary_data = {
        'scoring': scoring,
        'partition': partition,
        'initial_scores': initial_scores
    }

    return minimal_element, final_scores, auxiliary_data


if __name__ == "__main__":
    """ Load data """
    city_graph = ox.load_graphml('graph_skotniki.graphml')

    # For the drawing purposes I need the relabelling here
    relabel_nodes_mapping = {}
    for _num, _node in enumerate(city_graph.nodes):
        relabel_nodes_mapping[_node] = _num
    city_graph = nx.relabel_nodes(city_graph, relabel_nodes_mapping)
    colour_palette = list(mp.colors.BASE_COLORS.keys())

    """ If one wants to use a clustering method provided from other scripts """
    # clusters = pd.read_csv('skotniki_clusters.csv', index_col='Unnamed: 0')
    # clusters = clusters.to_dict()['0']
    # loaded_partition = [{k for k, v in clusters.items() if v == it} for it in range(5)]

    """ Crucial part - initiate this and update this data after the first run """
    graph_data = {
        'scoring': None,
        'partition': None, #loaded_partition - for the case of clustering 'from outside'
        'initial_scores': {}
    }
    removed_stops = []

    """ Calculations """
    for step in range(351):
        node_to_remove, node_scores, graph_data = community_centrality_node_removal(
            graph=city_graph,
            inactive_nodes=removed_stops,
            partition=graph_data['partition'], # if you want to recalculate the partition after each run, change to None
            scoring=graph_data['scoring'], # if you want to recalculate the scoring after each run, change to None
            preserve_original_partition=True, # if you want to recalculate the partition after each run, change to False
            preserve_original_scoring=False, # if you want to recalculate the scoring after each run, change to False
            partitioning_algorithm=nx.community.edge_current_flow_betweenness_partition,
            scoring_algorithm=nx.closeness_centrality,
            scoring_within_partitions=False, # calculate scoring separately for subgraphs determined by the clustering - connectivity within a community
            boost_neighbours=True, # if a node is removed, its neighbours (first and second) get some of its score
            boost_neighbours_factor=0.2, # a fraction that is moved to neighbours and square of that to the second neighbours,
            initial_scores=graph_data['initial_scores'], # store data to improve scores of neighbours of removed nodes
            partition_size_sensitivity=0.1, # a role that the size of a community plays in the final score relative to the connectivity
            number_of_sets=5
        )

        removed_stops += [node_to_remove]

        if step % 10 == 0:
            colour_list = np.zeros(len(city_graph.nodes))
            for _num, part in enumerate(graph_data['partition']):
                colour_list[list(part)] = _num
            colour_list = [colour_palette[int(t)] for t in colour_list]
            colour_list = ['white' if num in removed_stops else t for num, t in enumerate(colour_list)]
            node_sizes = [10 if num in removed_stops else 30 for num, t in enumerate(colour_list)]

            ox.plot_graph(city_graph, node_color=colour_list, node_size=node_sizes)

