from utils_topology import draw_bipartite_graph
import pickle
import networkx as nx
from netwulf import visualize
import json
import numpy as np
from utils_topology import draw_bipartite_graph


with open('data/results/18-05-22/rep_graphs_18-05-22.obj', 'rb') as file:
    e = pickle.load(file)

graph = e['bipartite_matching']
# visualize(graph, config=json.load(open('data/configs/netwulf_config.json')))
draw_bipartite_graph(graph, 1000, node_size=1, dpi=200, figsize=(10, 24))
