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

import Utils.utils_topology as utils
import NYC_tools.NYC_data_prep_functions as nyc_tools
from ExMAS.probabilistic_exmas import main as exmas_algo
from ExMAS.utils import make_graph as exmas_make_graph

with open("../critical_mass_data3.pickle", "rb") as file:
    data = pickle.load(file)

# o = []
# for _ra in range(100, 200, 10):
#     _o = []
#     for _add in range(0, 10, 1):
#         multiplier = 6/1000
#         key = "shared_ratio"
#         # _dat = [int(multiplier*t) for t in data[_ra+_add]]
#         _dat = [t for t in data[_ra + _add][key]]
#         _o.append(_dat)
#     o.append((np.mean(_o), np.std(_o)))

# o = []
# key = "Time_saved"
# multiplier = 6/1000
# rounding = False
# for _ra in range(100, 200, 10):
#     med = np.median([j for _add in range(10) for j in data[_ra + _add][key]])
#     feasible_range = (0.5*med, 1.5*med)
#     _dat = []
#     for _add in range(10):
#         mn = np.mean(data[_ra + _add][key])
#         if (mn > feasible_range[0]) and (mn < feasible_range[1]):
#             _dat.extend(data[_ra + _add][key])
#
#     if multiplier:
#         _dat = [t*multiplier for t in _dat]
#     if rounding:
#         _dat = [int(t) for t in _dat]
#     o.append((np.mean(_dat), np.std(_dat)))


# plt.errorbar(range(210, 410, 20), [t[0] for t in o], [t[1] for t in o],
#              ecolor="lightblue", elinewidth=1, capsize=4, linestyle='None', marker='o',
#              markerfacecolor="black", markeredgecolor="black")
# # plt.axline((210, [t[0] for t in o][0]), (290, [t[0] for t in o][4]), lw=0.7, ls=':', color="red", alpha=0.7)
# # plt.axline((290, [t[0] for t in o][4]), (390, [t[0] for t in o][9]), lw=0.7, ls=':', color="blue")
# plt.xticks(range(210, 410, 20))
# plt.savefig("critical_mass_2.jpeg", dpi=300)

o = []
key = "Time_saved"
multiplier = 6/1000
rounding = False
for _ra in range(100, 200, 10):
    for _add in range(10):
        o.append(np.mean(data[_ra + _add][key]))

plt.scatter(range(100, 200), o, s=5)
for mark in [147, 198]:
    plt.scatter([mark], o[mark-100], s=15, c='red')
plt.savefig('przykladne.jpeg', dpi=200)
