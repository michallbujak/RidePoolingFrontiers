import os

import networkx as nx
import collections
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scienceplots
from collections import Counter

print(os.getcwd())

import Utils.visualising_functions as vf
import Utils.utils_topology as utils
import ExMAS.utils as ut


# date = "18-01-23"
# special_name = "_net_fixed"
# sblts_exmas = "exmas"
#
#
#
# # with open('Topology/data/results/' + date + special_name + '/dotmap_list_' + date + '.obj', 'rb') as file:
# #     e = pickle.load(file)[31]
#
# with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Miscellaneous_scripts\data\22-03-24\147_homo_22-03-24.obj", 'rb') as file:
#     e = pickle.load(file)[0]
#
# # ut.plot_map_rides(e, [715], light=True, lw=3, fontsize=30, m_size=1)
# ut.plot_map_rides(e, [680], light=True, lw=3, fontsize=30, m_size=1)
#
#
# os.chdir(os.path.dirname(os.getcwd()))

# with open('Topology/data/results/' + date + special_name + '/dotmap_list_' + date + '.obj', 'rb') as file:
#     e = pickle.load(file)[31]

with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Miscellaneous_scripts\data\22-03-24\147_homo_22-03-24.obj", 'rb') as file:
    e = pickle.load(file)[0]

ut.plot_demand(e, dpi=300) #, origin_colour="green", destination_colour="orange")