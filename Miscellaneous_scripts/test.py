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
import statistics

with open(os.path.dirname(os.path.abspath(os.getcwd())) +
          "/Topology/data/single_graphs_example.obj", "rb") as file:
    G = pickle.load(file)["pairs_shareability"]


print(nx.katz_centrality_numpy(G))


