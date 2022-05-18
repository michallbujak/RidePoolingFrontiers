from utils_topology import draw_bipartite_graph
import pickle
import networkx as nx
from netwulf import visualize
import json
import numpy as np
from matplotlib import pyplot as plt
from utils_topology import concat_all_graph_list
from utils_topology import analyse_concatenated_all_graph_list

with open('data/results/18-05-22/all_graphs_list.obj', 'rb') as file:
    e = pickle.load(file)

z = concat_all_graph_list(e)
y = analyse_concatenated_all_graph_list(z)

print(round(y, 3))
# visualize(G1, config=json.load(open('data/configs/netwulf_config.json')))
