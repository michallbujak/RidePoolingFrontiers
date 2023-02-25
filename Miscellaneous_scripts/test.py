import os

import networkx as nx
import collections
import numpy as np
import pickle
import matplotlib.pyplot as plt
import Utils.visualising_functions as vf
import Utils.utils_topology as utils
import ExMAS.utils as ut
import pandas as pd
import seaborn as sns
import scienceplots
from collections import Counter


with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\18-01-23_net_fixed\all_graphs_list_18-01-23.obj", 'rb') as file:
    all_graphs = pickle.load(file)

with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\18-01-23_net_fixed\rep_graphs_18-01-23.obj", 'rb') as file:
    rep_graphs = pickle.load(file)

topological_config = utils.get_parameters(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\configs\topology_settings_no_random2.json")
utils.create_results_directory(topological_config)
#
topological_config.path_results = "C:/Users/szmat/Documents/GitHub/ExMAS_sideline/Topology/data/results/18-01-23_net_fixed/"

class TempClass:
    def __init__(self, dataset: pd.DataFrame, output_path: str, output_temp: str, input_variables: list,
                 all_graph_properties: list, kpis: list, graph_properties_to_plot: list, labels: dict,
                 err_style: str = "band", date: str = '000'):
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
        if 'Replication_ID' in dataset.columns or 'Replication' in dataset.columns:
            for x in ['Replication_ID', 'Replication']:
                if x in dataset.columns:
                    self.dataset = dataset.drop(columns=[x])
                else:
                    pass
        else:
            self.dataset = dataset
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
        self.date = date

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
        self.dataset['UtilityGained'] = (self.dataset['PassUtility_ns'] - self.dataset['PassUtility']) / \
                                        self.dataset['PassUtility_ns']
        self.dataset['Fraction_isolated'] = self.dataset['No_isolated_pairs'] / self.dataset['nP']
        self.dataset_grouped = self.dataset.groupby(self.input_variables)

    def plot_kpis_properties(self):
        plot_arguments = [(x, y) for x in self.graph_properties_to_plot for y in self.kpis]
        dataset = self.dataset.copy()
        binning = False
        for counter, value in enumerate(self.input_variables):
            min_val = min(self.dataset[value])
            max_val = max(self.dataset[value])
            if min_val == 0 and max_val == 0:
                binning = False
            else:
                step = (max_val - min_val) / 3
                if min_val < 5:
                    bins = np.round(np.append(np.arange(min_val * 0.98, max_val * 1.02, step), [max_val + step]), 3)
                    bins = np.round(np.arange(0, 0.51, 0.1), 3)
                else:
                    bins = np.round(np.append(np.arange(min_val * 0.98, max_val * 1.02, step), [max_val + step]), 0)
                labels = [f'{i}+' if j == np.inf else f'{i}-{j}' for i, j in
                          zip(bins, bins[1:])]  # additional part with infinity
                dataset[self.labels[value] + " bin"] = pd.cut(dataset[value], bins, labels=labels)
                binning = True

        for counter, j in enumerate(plot_arguments):
            if not binning:
                fig, ax = plt.subplots()
                sns.scatterplot(x=j[0], y=j[1], data=dataset, palette="crest")
                ax.set_xlabel(self.labels[j[0]])
                ax.set_ylabel(self.labels[j[1]])
                plt.savefig(self.output_temp + 'kpis_properties_' + str(counter) + '.png')
                plt.close()
            else:
                if len(self.input_variables) == 1:
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=j[0], y=j[1], data=dataset,
                                    hue=dataset[self.labels[self.input_variables[0]] + " bin"], palette="crest")
                    ax.set_xlabel(self.labels[j[0]])
                    ax.set_ylabel(self.labels[j[1]])
                    if counter == 8:
                        plt.legend(fontsize=8, title="Sharing discount", title_fontsize=9)
                    else:
                        ax.get_legend().remove()
                    # ax.get_legend().remove()
                    plt.savefig(self.output_temp + j[0] + j[1] + '_v2.png')
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
    def do(self):
        self.alternate_kpis()
        self.plot_kpis_properties()


variables = ['shared_discount']
TempClass(pd.read_excel(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\18-01-23_net_fixed\merged_files_18-01-23.xlsx"),
          topological_config.path_results,
          topological_config.path_results + "temp/",
          variables,
          topological_config.graph_topological_properties,
          topological_config.kpis,
          topological_config.graph_properties_against_inputs,
          topological_config.dictionary_variables).do()
