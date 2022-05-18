from utils_topology import draw_bipartite_graph
import pickle
import networkx as nx
from netwulf import visualize
import json
import numpy as np


with open('data/results/18-05-22/all_graphs_list.obj', 'rb') as file:
    e = pickle.load(file)

graph = e['pairs_matching']
visualize(graph, config=json.load(open('data/configs/netwulf_config.json')))
