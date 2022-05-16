import pickle
import networkx as nx
from netwulf import visualize

with open('data/results/16-05-22/rep_graphs.obj', 'rb') as file:
    e = pickle.load(file)

graph = e['pairs_shareability']
visualize(graph)

