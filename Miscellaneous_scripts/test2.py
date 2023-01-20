import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.ticker as mtick
import os
import scienceplots
import networkx as nx
import collections
import numpy as np
import time

import scienceplots

plt.style.use(['science', 'no-latex'])
os.chdir(os.path.dirname(os.getcwd()))

with open('res_bip.obj', 'rb') as file:
    res_bip = pickle.load(file)

with open('res_sim.obj', 'rb') as file:
    res_sim = pickle.load(file)

name = "clustering"

x = [e["disc"] for e in res_bip]
y1 = [e["average_" + name + "_group1"] for e in res_bip]
y2 = [e["average_" + name + "_group0"] for e in res_bip]

#
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(x, y1, 'g-')
# ax2.plot(x, y2, label='b-')
# ax1.set_xlabel('Sharing discount')
# plt.xlim(left=0, right=0.5)
# ax1.set_ylabel("Av. " + name + " travellers", color='g')
# ax2.set_ylabel("Av. " + name + " rides", color='b')
# plt.savefig("Topology/data/results/18-01-23_net_fixed/figs/" + name + "_discount")
# plt.close()

x = [e["disc"] for e in res_sim]
y1 = [e["average_degree"] for e in res_sim]
y2 = [e["average_clustering"] for e in res_sim]

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(x, y1, 'g-')
# ax2.plot(x, y2, label='b-')
# ax1.set_xlabel('Sharing discount')
# plt.xlim(left=0, right=0.5)
# ax1.set_ylabel('Av. degree', color='g')
# ax2.set_ylabel('Av. clustering', color='b')
# plt.savefig("Topology/data/results/18-01-23_net_fixed/figs/simple_both_discount")
# plt.close()

e = 0


