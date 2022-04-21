import json
from dotmap import DotMap
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import collections
import logging
import sys


def get_parameters(path, time_correction=False):
    with open(path) as json_file:
        data = json.load(json_file)
        config = DotMap(data)
    if time_correction:
        config['t0'] = pd.Timestamp('15:00')

    return config


def init_log(logger_level, logger=None):
    if logger_level == 'DEBUG':
        level = logging.DEBUG
    elif logger_level == 'WARNING':
        level = logging.WARNING
    elif logger_level == 'CRITICAL':
        level = logging.CRITICAL
    elif logger_level == 'INFO':
        level = logging.INFO
    else:
        raise Exception("Not accepted logger level, please choose: 'DEBUG', 'WARNING', 'CRITICAL', 'INFO'")
    if logger is None:
        logging.basicConfig(stream=sys.stdout, format='%(asctime)s-%(levelname)s-%(message)s',
                            datefmt='%H:%M:%S', level=level)

        logger = logging.getLogger()

        logger.setLevel(level)
        return logging.getLogger(__name__)
    else:
        logger.setLevel(level)
        return logger


class GraphStatistics:
    def __init__(self, graph, logging_level="INFO"):
        self.logger = init_log(logging_level)
        self.G = graph
        self.connected = nx.is_connected(self.G)
        self.bipartite = nx.is_bipartite(self.G)
        self.average_degree = None
        self.maximum_degree = None
        self.average_clustering_coefficient = None
        self.average_clustering_group0 = None
        self.average_clustering_group1 = None
        self.components = None
        self.proportion_max_component = None
        self.num_nodes_group0 = None
        self.num_nodes_group1 = None
        self.average_degree_group0 = None
        self.average_degree_group1 = None
        self.number_of_isolated_pairs = None
        self.average_clustering_group0_reduced = None
        self.average_clustering_group1_reduced = None
        # Objects to be stored rather than strict output
        self.group0_colour = None
        self.group1_colour = None
        self.reduced_graph = None
        self.group0_colour_reduced = None
        self.group1_colour_reduced = None

    def initial_analysis(self):
        self.logger.info('Graph is connected: {}'.format(self.connected))
        self.logger.info('Graph is bipartite: {}'.format(self.bipartite))
        self.logger.info('Number of nodes: {}'.format(self.G.number_of_nodes()))
        self.logger.info('Number of edges: {}'.format(self.G.number_of_edges()))
        if self.bipartite:
            self.colouring_graph()

    def colouring_graph(self):
        if self.bipartite:
            partition_for_bipartite = nx.bipartite.basic.color(self.G)
            for colour_key in partition_for_bipartite.keys():
                self.G.nodes[colour_key]['bipartite'] = partition_for_bipartite[colour_key]
            total_colouring = {k: v['bipartite'] for k, v in self.G._node.copy().items()}
            self.group0_colour = {k: v for k, v in total_colouring.items() if v == 0}
            self.group1_colour = {k: v for k, v in total_colouring.items() if v == 1}
            # Group 0 shall be longer
            if len(self.group0_colour) > len(self.group1_colour):
                pass
            else:
                self.group0_colour, self.group1_colour = self.group1_colour, self.group0_colour

            # Additional analysis removing rides of degree 1
            remove_bc_of_degree = [node for node, degree in dict(self.G.degree()).items() if degree == 1]
            remove_only_from_group0 = [node for node in remove_bc_of_degree if node in self.group0_colour.keys()]
            self.reduced_graph = self.G.copy()
            self.reduced_graph.remove_nodes_from(remove_only_from_group0)
            partition_for_bipartite = nx.bipartite.basic.color(self.reduced_graph)
            for colour_key in partition_for_bipartite.keys():
                self.reduced_graph.nodes[colour_key]['bipartite'] = partition_for_bipartite[colour_key]
            total_colouring = {k: v['bipartite'] for k, v in self.reduced_graph._node.copy().items()}
            self.group0_colour_reduced = {k: v for k, v in total_colouring.items() if v == 0}
            self.group1_colour_reduced = {k: v for k, v in total_colouring.items() if v == 1}
        else:
            pass

    def degree_distribution(self, degree_histogram=False, degree_cdf=False):
        degree_sequence = sorted([d for n, d in self.G.degree()], reverse=False)
        degree_counter = collections.Counter(degree_sequence)
        deg, cnt = zip(*degree_counter.items())
        self.average_degree = np.sum(np.multiply(deg, cnt)) / self.G.number_of_nodes()
        self.maximum_degree = max(degree_sequence)
        self.logger.info('Average degree: {}'.format(self.average_degree))
        self.logger.info('Maximum degree: {}'.format(self.maximum_degree))

        if self.bipartite:
            degrees = dict(self.G.degree())
            group0 = {k: v for k, v in degrees.items() if k in self.group0_colour.keys()}
            group1 = {k: v for k, v in degrees.items() if k in self.group1_colour.keys()}
            self.average_degree_group0 = sum(group0.values()) / len(group0)
            self.average_degree_group1 = sum(group1.values()) / len(group1)

        if degree_histogram:
            plt.bar(*np.unique(degree_sequence, return_counts=True))
            plt.title("Degree histogram")
            plt.xlabel("Degree")
            plt.ylabel("# of Nodes")
            plt.show()

        if degree_cdf:
            cs = np.cumsum(cnt)
            n = len(degree_sequence)
            plt.style.use('seaborn-whitegrid')
            plt.plot(sorted(deg), cs / n, 'bo', linestyle='-', linewidth=1.2, markersize=2.5)
            plt.title("True Cumulative Distribution plot")
            plt.axhline(y=0.9, color='r', linestyle='dotted', alpha=0.5, label='0.9')
            plt.ylabel("P(k>=Degree)")
            plt.xlabel("Degree")
            plt.xlim(0, max(degree_sequence))
            plt.ylim((cs / n)[0], 1.05)
            plt.show()

    def nodes_per_colour(self):
        if self.bipartite:
            self.num_nodes_group0 = len(self.group0_colour)
            self.num_nodes_group1 = len(self.group1_colour)
        else:
            pass

    def clustering_coefficient(self, detailed=False):
        if not self.bipartite:
            self.logger.info('The graph is not bipartite, hence the clustering coefficient is based on triangles.')
            self.average_clustering_coefficient = nx.average_clustering(self.G)
            self.logger.info("Graph's average clustering coefficient is {}.".format(self.average_clustering_coefficient))
            if detailed:
                self.logger.info('Clustering coefficients per node: \n', nx.clustering(self.G))
                self.logger.info('Transitivity per node: \n', nx.transitivity(self.G))
                self.logger.info('Triangles per node: \n', nx.triangles(self.G))
        else:
            self.logger.info('The graph is bipartite, hence the clustering coefficient in based on squares.')
            sq_coefficient = nx.square_clustering(self.G)
            group0 = {k: v for k, v in sq_coefficient.items() if k in self.group0_colour.keys()}
            group1 = {k: v for k, v in sq_coefficient.items() if k in self.group1_colour.keys()}
            if len(sq_coefficient) != 0:
                self.average_clustering_coefficient = sum(sq_coefficient.values()) / len(sq_coefficient)
            else:
                self.average_clustering_coefficient = 0
            self.average_clustering_group0 = sum(group0.values()) / len(group0)
            self.average_clustering_group1 = sum(group1.values()) / len(group1)
            self.logger.info('Average clustering coefficient: ', self.average_clustering_coefficient)
            self.logger.info('Average clustering coefficient in group 0: ', self.average_clustering_group0)
            self.logger.info('Average clustering coefficient in group 1: ', self.average_clustering_group1)

            # Reduced graphs by nodes in group1 whose degree is equal to 1
            sq_coefficient = nx.square_clustering(self.reduced_graph)
            group0 = {k: v for k, v in sq_coefficient.items() if k in self.group0_colour_reduced.keys()}
            group1 = {k: v for k, v in sq_coefficient.items() if k in self.group1_colour_reduced.keys()}
            if len(group0) != 0:
                self.average_clustering_group0_reduced = sum(group0.values()) / len(group0)
            else:
                self.average_clustering_group0_reduced = 0
            if len(group1) != 0:
                self.average_clustering_group1_reduced = sum(group1.values()) / len(group1)
            else:
                self.average_clustering_group1_reduced = 0

    def component_analysis(self, plot=False):
        g_components = list(nx.connected_components(self.G))
        g_components.sort(key=len, reverse=True)
        self.components = g_components
        self.logger.info('Number of connected components: ', len(self.components))
        self.logger.info('Sizes of the components: ', [len(i) for i in self.components])
        self.proportion_max_component = len(self.components[0]) / self.G.number_of_nodes()
        self.number_of_isolated_pairs = sum(1 if x == 2 else 0 for x in [len(i) for i in self.components])
        if plot:
            plt.style.use('seaborn-whitegrid')
            plt.bar(range(len(self.components)), [len(i) for i in self.components])
            plt.title("Sorted sizes of connected components")
            plt.ylabel("No. of nodes")
            plt.xlabel("Component's ID")
            plt.xticks(range(len(self.components)))
            plt.show()

    def all_analysis(self, degree_distribution=False, degree_cdf=False, detailed_clustering=False,
                     plot_components=False):
        GraphStatistics.initial_analysis(self)
        GraphStatistics.colouring_graph(self)
        GraphStatistics.nodes_per_colour(self)
        GraphStatistics.degree_distribution(self, degree_distribution, degree_cdf)
        GraphStatistics.clustering_coefficient(self, detailed_clustering)
        GraphStatistics.component_analysis(self, plot_components)


def worker_topological_properties(GraphStatObj):
    data_output = pd.DataFrame()
    GraphStatObj.all_analysis()
    if GraphStatObj.bipartite:
        data_output = data_output.append([GraphStatObj.num_nodes_group0, GraphStatObj.num_nodes_group1,
                                          GraphStatObj.average_degree, GraphStatObj.maximum_degree,
                                          GraphStatObj.average_degree_group0,
                                          GraphStatObj.average_degree_group1,
                                          GraphStatObj.average_clustering_coefficient,
                                          GraphStatObj.proportion_max_component, len(GraphStatObj.components),
                                          GraphStatObj.average_clustering_group0,
                                          GraphStatObj.average_clustering_group1,
                                          GraphStatObj.number_of_isolated_pairs,
                                          GraphStatObj.average_clustering_group0_reduced,
                                          GraphStatObj.average_clustering_group1_reduced])
        data_output.index = ['No_nodes_group0', 'No_nodes_group1', 'Average_degree',
                             'Maximum_degree', 'Average_degree_group0', 'Average_degree_group1',
                             'Avg_clustering',
                             'Proportion_max_component', 'No_components', 'Average_clustering_group0',
                             'Average_clustering_group1', 'No_isolated_pairs',
                             'Average_clustering_group0_reduced',
                             'Average_clustering_group1_reduced']
    else:
        data_output = data_output.append([GraphStatObj.average_degree, GraphStatObj.maximum_degree,
                                          GraphStatObj.average_clustering_coefficient,
                                          GraphStatObj.proportion_max_component,
                                          len(GraphStatObj.components)])
        data_output.index = ['Average_degree', 'Maximum_degree', 'Avg_clustering',
                             'Proportion_max_component', 'No. of components']

    return data_output
