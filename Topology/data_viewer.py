import pickle
import networkx as nx
from netwulf import visualize
import json
import numpy as np
from utils_topology import draw_bipartite_graph
import utils_topology as utils
import matplotlib.pyplot as plt
import datetime

# with open('data/results/20-05-22/dotmap_list_20-05-22.obj', 'rb') as file:
#     e = pickle.load(file)
#
# topological_config = utils.get_parameters('data/configs/topology_settings.json')
# utils.create_results_directory(topological_config)
#
# num_list = list(range(100, 1000, 100))
# for num in num_list:
#     draw_bipartite_graph(utils.analyse_edge_count(e[:num], topological_config,
#                                                   list_types_of_graph=['bipartite_matching'], logger_level='WARNING')[
#                              'bipartite_matching'],
#                          num, node_size=1, dpi=200, figsize=(10, 24), plot=False,
#                          config=topological_config, save=True, saving_number=num)

num_list = [1] + list(range(100, 1000, 100))

topological_config = utils.get_parameters('data/configs/topology_settings.json')
utils.create_results_directory(topological_config)

fig, axes = plt.subplots(nrows=2, ncols=5, sharex='col', sharey='row')

for col in range(2):
    for row in range(5):
        num = num_list[5*col+row]
        axes[col, row].imshow(topological_config.path_results + "temp/graph_" + str(datetime.date.today().strftime("%d-%m-%y"))
                    + "_no_" + num + ".png")

plt.show()



