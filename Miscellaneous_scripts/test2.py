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
from netwulf import visualize
import json

os.chdir(os.path.dirname(os.getcwd()))

date = "19-01-23"
special_name = "_full"
sblts_exmas = "exmas"

with open('Topology/data/results/' + date + special_name + '/rep_graphs_' + date + '.obj', 'rb') as file:
    e = pickle.load(file)

# with open('data/results/' + date + special_name + '/dotmap_list_' + date + '.obj', 'rb') as file:
#     e = pickle.load(file)

# with open('data/results/' + date + '/all_graphs_list_' + date + '.obj', 'rb') as file:
#     e = pickle.load(file)

topological_config = utils.get_parameters('Topology/data/configs/topology_settings_panel.json')
# utils.create_results_directory(topological_config, date=date)
topological_config.path_results = 'Topology/data/results/' + date + special_name + '/'




# visualize(e['pairs_matching'], config=json.load(open('Topology/data/configs/netwulf_config.json')))
vf.draw_bipartite_graph(e['bipartite_matching'], 1000, topological_config, date=date, save=True,
                     name="full_bi_match", dpi=300, colour_specific_node=74, node_size=1,
                     default_edge_size=0.2, emphasize_coloured_node=5, alpha=None)

# vf.overwrite_netwulf(e['pairs_shareability'], topological_config, 74, alpha=0.3, netwulf_config=json.load(open('Topology/data/configs/netwulf_config2.json')),
#                      save_name="pairs_share", node_size=7)
