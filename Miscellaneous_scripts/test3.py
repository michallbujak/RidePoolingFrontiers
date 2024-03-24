import pandas as pd
import multiprocessing as mp
import datetime
from netwulf import visualize
import pickle
import networkx as nx
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

# with open("../sensitivity_classes.pickle", "rb") as file:
#     data = pickle.load(file)
#
# o = []
# for key in data.keys():
#     prob_data = data[key][0]
#     _o = []
#     for num in range(len(prob_data)):
#         tmp_data = prob_data[num]['exmas']['res']
#         _o.append(6*(tmp_data["VehHourTrav_ns"] - tmp_data["VehHourTrav"])/1000)
#
#     o.append((np.mean(_o), np.std(_o)))
#
# plt.errorbar(np.arange(0, 0.42, 0.03), [t[0] for t in o], [t[1] for t in o],
#              ecolor="lightblue", elinewidth=1, capsize=4, linestyle='None', marker='o',
#              markerfacecolor="black", markeredgecolor="black")
# plt.savefig('sensitivity_classes.jpeg', dpi=300)





