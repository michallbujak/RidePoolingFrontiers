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

date = "18-01-23"
special_name = "_net_fixed"
sblts_exmas = "exmas"

os.chdir(os.path.dirname(os.getcwd()))

# with open('Topology/data/results/' + date + special_name + '/dotmap_list_' + date + '.obj', 'rb') as file:
#     e = pickle.load(file)[31]

with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Miscellaneous_scripts\data\14-03-23\dotmap_list_14-03-23.obj", 'rb') as file:
    e = pickle.load(file)[0]

with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Miscellaneous_scripts\data\14-03-23\config_14-03-23.obj", 'rb') as file:
    e2 = pickle.load(file)[0]

ut.plot_demand(e, e2)