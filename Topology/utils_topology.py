import json
from dotmap import DotMap
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import collections
import logging
import sys
import seaborn as sns


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


def alternate_kpis(dataset):
    if 'nP' in dataset.columns:
        pass
    else:
        dataset['nP'] = dataset['No_nodes_group1']

    dataset['Proportion_singles'] = dataset['SINGLE'] / dataset['nP']
    dataset['Proportion_pairs'] = dataset['PAIRS'] / dataset['nP']
    dataset['Proportion_triples'] = dataset['TRIPLES'] / dataset['nP']
    dataset['Proportion_triples_plus'] = (dataset['nP'] - dataset['SINGLE'] -
                                               dataset['PAIRS']) / dataset['nP']
    dataset['Proportion_quadruples'] = dataset['QUADRIPLES'] / dataset['nP']
    dataset['Proportion_quintets'] = dataset['QUINTETS'] / dataset['nP']
    dataset['Proportion_six_plus'] = dataset['PLUS5'] / dataset['nP']
    dataset['SavedVehHours'] = (dataset['VehHourTrav_ns'] - dataset['VehHourTrav']) / \
                                    dataset['VehHourTrav_ns']
    dataset['AddedPasHours'] = (dataset['PassHourTrav'] - dataset['PassHourTrav_ns']) / \
                                    dataset['PassHourTrav_ns']
    dataset['UtilityGained'] = (dataset['PassUtility'] - dataset['PassUtility_ns']) / \
                                    dataset['PassUtility_ns']
    dataset['Fraction_isolated'] = dataset['No_isolated_pairs']/dataset['nP']
    return dataset


def amend_merged_file(merged_file, alter_kpis=False, inplace=True):
    if not inplace:
        merged_file = merged_file.copy(deep=True)
    merged_file.drop(columns=['_typ', 'dtype'], inplace=True)
    merged_file.reset_index(inplace=True, drop=True)
    if alter_kpis:
        merged_file = alternate_kpis(merged_file)
    return merged_file


def merge_results(dotmaps_list_results, topo_dataframes, settings_list):
    res = [pd.concat([z, x, y.sblts.res]) for z, x, y in
           zip([pd.Series(k) for k in settings_list], topo_dataframes, dotmaps_list_results)]
    merged_file = pd.DataFrame()
    for item in res:
        merged_file = pd.concat([merged_file, item.T])
    amend_merged_file(merged_file)
    return merged_file


class APosterioriAnalysis:
    def __init__(self, dataset: pd.DataFrame, output_path: str, output_temp: str, input_variables: list,
                 all_graph_properties: list, kpis: list, graph_properties_to_plot: list, labels: dict,
                 err_style: str = "band"):
        """
        Class designed to performed analysis on merged results from shareability graph properties.
        :param dataset: input merged datasets from replications
        :param output_path: output for final results
        :param output_temp: output for temporal results required in the process
        :param input_variables: search space variables
        :param all_graph_properties: all graph properties for heatmap/correlation analysis
        :param kpis: final matching coefficients to take into account
        :param graph_properties_to_plot: properties of graph to be plotted
        :param labels: dictionary of labels
        :param err_style: for line plots style of the error
        """
        self.dataset = dataset.drop(columns=['Replication_ID'])
        self.input_variables = input_variables
        self.all_graph_properties = all_graph_properties
        self.dataset_grouped = self.dataset.groupby(self.input_variables)
        self.output_path = output_path
        self.output_temp = output_temp
        self.kpis = kpis
        self.graph_properties_to_plot = graph_properties_to_plot
        self.labels = labels
        self.err_style = err_style
        self.heatmap = None

    def alternate_kpis(self):
        if 'nP' in self.dataset.columns:
            pass
        else:
            self.dataset['nP'] = self.dataset['No_nodes_group1']

        self.dataset['Proportion_singles'] = self.dataset['SINGLE'] / self.dataset['nR']
        self.dataset['Proportion_pairs'] = self.dataset['PAIRS'] / self.dataset['nR']
        self.dataset['Proportion_triples'] = self.dataset['TRIPLES'] / self.dataset['nR']
        self.dataset['Proportion_triples_plus'] = (self.dataset['nR'] - self.dataset['SINGLE'] -
                                                   self.dataset['PAIRS']) / self.dataset['nR']
        self.dataset['Proportion_quadruples'] = self.dataset['QUADRIPLES'] / self.dataset['nR']
        self.dataset['Proportion_quintets'] = self.dataset['QUINTETS'] / self.dataset['nR']
        self.dataset['Proportion_six_plus'] = self.dataset['PLUS5'] / self.dataset['nR']
        self.dataset['SavedVehHours'] = (self.dataset['VehHourTrav_ns'] - self.dataset['VehHourTrav']) / \
                                        self.dataset['VehHourTrav_ns']
        self.dataset['AddedPasHours'] = (self.dataset['PassHourTrav'] - self.dataset['PassHourTrav_ns']) / \
                                        self.dataset['PassHourTrav_ns']
        self.dataset['UtilityGained'] = (self.dataset['PassUtility'] - self.dataset['PassUtility_ns']) / \
                                        self.dataset['PassUtility_ns']
        self.dataset['Fraction_isolated'] = self.dataset['No_isolated_pairs'] / self.dataset['nP']
        self.dataset_grouped = self.dataset.groupby(self.input_variables)

    def boxplot_inputs(self):
        for counter, y_axis in enumerate(self.all_graph_properties):
            dataset = self.dataset.copy()
            if len(self.input_variables) <= 2:
                if len(self.input_variables) == 1:
                    sns.boxplot(x=self.input_variables[0], y=y_axis, data=dataset) \
                        .set(xlabel=self.labels[self.input_variables[0]], ylabel=self.labels[y_axis])
                elif len(self.input_variables) == 2:
                    temp_dataset = dataset.copy()
                    temp_dataset[self.labels[self.input_variables[1]]] = temp_dataset[self.input_variables[1]]
                    sns.boxplot(x=self.input_variables[0], y=y_axis, data=temp_dataset,
                                hue=self.labels[self.input_variables[1]]) \
                        .set(xlabel=self.labels[self.input_variables[0]], ylabel=self.labels[y_axis])
                else:
                    break
                plt.savefig(self.output_temp + 'temp_boxplot_' + str(counter) + '.png')
                plt.close()
            else:
                raise Exception('Grouping variables number is:', len(self.input_variables), ' - too long for boxplot.')

    def line_plot_inputs(self):
        for counter, x_axis in enumerate(self.input_variables):
            if len(self.graph_properties_to_plot) <= 2:
                if len(self.input_variables) == 1:
                    sns.lineplot(x=x_axis, y=self.graph_properties_to_plot[0], data=self.dataset)
                    plt.xlabel(self.labels[x_axis])
                    plt.ylabel(self.labels[self.graph_properties_to_plot[0]], color='b')
                elif len(self.input_variables) == 2:
                    fig, ax1 = plt.subplots()
                    ax2 = ax1.twinx()
                    sns.lineplot(x=x_axis, y=self.graph_properties_to_plot[0], data=self.dataset, color='b', ax=ax1,
                                 err_style=self.err_style)
                    sns.lineplot(x=x_axis, y=self.graph_properties_to_plot[1], data=self.dataset, color='r', ax=ax2,
                                 err_style=self.err_style)
                    ax1.set_xlabel(self.labels[x_axis])
                    ax1.set_ylabel(self.labels[self.graph_properties_to_plot[0]], color='b')
                    ax2.set_ylabel(self.labels[self.graph_properties_to_plot[1]], color='r')
                else:
                    pass
                plt.savefig(self.output_temp + 'temp_lineplot_' + str(counter) + '.png')
                plt.close()
            else:
                raise Exception('Grouping variables number is:', len(self.input_variables), ' - too long for lineplot.')

    def plot_kpis_properties(self):
        plot_arguments = [(x, y) for x in self.graph_properties_to_plot for y in self.kpis]
        dataset = self.dataset.copy()
        for counter, value in enumerate(self.input_variables):
            min_val = min(self.dataset[value])
            max_val = max(self.dataset[value])
            step = (max_val - min_val) / 3
            if min_val < 5:
                bins = np.round(np.append(np.arange(min_val * 0.98, max_val * 1.02, step), [max_val + step]), 3)
            else:
                bins = np.round(np.append(np.arange(min_val * 0.98, max_val * 1.02, step), [max_val + step]), 0)
            labels = [f'{i}+' if j == np.inf else f'{i}-{j}' for i, j in
                      zip(bins, bins[1:])]  # additional part with infinity
            dataset[self.labels[value] + " bin"] = pd.cut(dataset[value], bins, labels=labels)

        for counter, j in enumerate(plot_arguments):
            if len(self.input_variables) == 1:
                fig, ax = plt.subplots()
                sns.scatterplot(x=j[0], y=j[1], data=dataset, hue=dataset[self.labels[self.input_variables[0]] + " bin"]
                                , palette="crest")
                ax.set_xlabel(self.labels[j[0]])
                ax.set_ylabel(self.labels[j[1]])
                plt.savefig(self.output_temp + 'kpis_properties_' + str(counter) + '.png')
                plt.close()
            elif len(self.input_variables) == 2:
                fix, ax = plt.subplots()
                sns.scatterplot(x=j[0], y=j[1], data=dataset,
                                hue=dataset[self.labels[self.input_variables[0]] + " bin"],
                                size=dataset[self.labels[self.input_variables[1]] + " bin"], palette="crest")
                ax.set_xlabel(self.labels[j[0]])
                ax.set_ylabel(self.labels[j[1]])
                plt.savefig(self.output_temp + 'kpis_properties_' + str(counter) + '.png')
                plt.close()
            else:
                fig, ax = plt.subplots()
                sns.scatterplot(x=j[0], y=j[1], data=dataset, palette="crest")
                ax.set_xlabel(self.labels[j[0]])
                ax.set_ylabel(self.labels[j[1]])
                plt.savefig(self.output_temp + 'kpis_properties_' + str(counter) + '.png')
                plt.close()

    def create_heatmap(self):
        df = self.dataset[self.all_graph_properties + self.kpis]
        for column in df.columns:
            df.rename(columns={column: self.labels[column]}, inplace=True)

        corr = df.corr()
        self.heatmap = corr
        plt.figure(figsize=(10, 10))
        sns.heatmap(corr,
                    xticklabels=corr.columns,
                    yticklabels=corr.columns,
                    linewidths=.5,
                    annot=True,
                    center=0,
                    cmap='OrRd')
        plt.subplots_adjust(bottom=0.3, left=0.3)
        plt.savefig(self.output_temp + 'heatmap' + '.png')
        plt.close()
        self.heatmap = round(self.heatmap, 3).style.background_gradient(cmap='coolwarm').set_precision(2)

    def save_grouped_results(self):
        writer = pd.ExcelWriter(self.output_path + 'Final_results_' + '_'.join(self.input_variables) + '.xlsx',
                                engine='xlsxwriter')
        self.dataset_grouped.min().to_excel(writer, sheet_name='Min')
        self.dataset_grouped.mean().to_excel(writer, sheet_name='Mean')
        self.dataset_grouped.max().to_excel(writer, sheet_name='Max')
        workbook = writer.book

        worksheet = workbook.add_worksheet('Boxplots')
        for counter in range(len(self.all_graph_properties)):
            worksheet.insert_image('B' + str(counter * 25 + 1),
                                   self.output_temp + 'temp_boxplot_' + str(counter) + '.png')

        worksheet = workbook.add_worksheet('Lineplots')
        for counter in range(len(self.graph_properties_to_plot)):
            worksheet.insert_image('B' + str(counter * 25 + 1),
                                   self.output_temp + 'temp_lineplot_' + str(counter) + '.png')

        worksheet = workbook.add_worksheet('KpiPlots')
        for counter in range(len(self.graph_properties_to_plot) * len(self.kpis)):
            worksheet.insert_image('B' + str(counter * 25 + 1),
                                   self.output_temp + 'kpis_properties_' + str(counter) + '.png')

        self.heatmap.to_excel(writer, sheet_name='Correlation')
        worksheet = workbook.get_worksheet_by_name('Correlation')
        worksheet.insert_image('B' + str(len(self.all_graph_properties) * 2 + 5), self.output_temp + 'heatmap' + '.png')
        writer.save()

    def do_all(self):
        self.alternate_kpis()
        self.line_plot_inputs()
        self.boxplot_inputs()
        self.plot_kpis_properties()
        self.create_heatmap()
        self.save_grouped_results()
