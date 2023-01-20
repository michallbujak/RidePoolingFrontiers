import networkx as nx
import collections
import numpy as np
import pickle
import matplotlib.pyplot as plt
import Utils.visualising_functions as vf
import Utils.utils_topology as utils


# with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\19-01-23\all_graphs_list_19-01-23.obj", 'rb') as file:
#     all_graphs = pickle.load(file)

with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\19-01-23\rep_graphs_19-01-23.obj", 'rb') as file:
    rep_graphs = pickle.load(file)

topological_config = utils.get_parameters(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\configs\topology_settings_panel.json")
utils.create_results_directory(topological_config)

topological_config.path_results = "C:/Users/szmat/Documents/GitHub/ExMAS_sideline/Topology/data/results/19-01-23/"

vf.draw_bipartite_graph(rep_graphs["bipartite_matching"], 1000, topological_config, True)
