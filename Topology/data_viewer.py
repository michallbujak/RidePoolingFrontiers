import pickle
import networkx as nx
from netwulf import visualize

with open('rep_graphs_1000.obj', 'rb') as file:
    e = pickle.load(file)

graph = e['pairs_shareability']
visualize(graph)