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

with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\25-11-22_baseline\dotmap_list_25-11-22.obj", 'rb') as file:
    all_graphs = pickle.load(file)


e = 0
