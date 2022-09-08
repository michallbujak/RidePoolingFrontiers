import pandas as pd
import numpy as np
import pickle
import scipy.stats as ss
import Topology.utils_topology as utils
from test2 import func
import os
import sys
import networkx as nx
import Topology.graph_properties_class as test_func

with open(os.path.dirname(os.path.abspath(os.getcwd())) +
          "/Topology/data/single_graphs_example.obj", "rb") as file:
    G = pickle.load(file)["pairs_shareability"]

second_neighbours = test_func.nodes_neighbours(G, 2)
r = {key: len(val) for key, val in second_neighbours.items()}
q = {node: sum([r[t] for t in G.neighbors(node)]) for node in G.nodes}
x = {node: sum([q[t] for t in G.neighbors(node)]) for node in G.nodes}
z = 0
