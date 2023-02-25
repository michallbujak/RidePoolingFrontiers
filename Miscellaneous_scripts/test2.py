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

plt.style.use(['science', 'no-latex'])

df = pd.read_excel(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\19-01-23_full\frame_evolution_19-01-23.xlsx")

plt, ax1 = plt.subplots(figsize=(4, 3), dpi=200)
ax2 = ax1.twinx()
# ax1.plot(df["average_degree"], label="Average degree", color='blue', lw=.5, marker='o', linestyle='solid', markersize=1)
# ax2.plot(df["max_comp"], label="Greatest component", color='red', lw=.5, marker='o', linestyle='solid', markersize=1)
ax1.plot(df["average_degree"], label="Average degree", color='blue', lw=1)
ax2.plot(df["max_comp"], label="Largest component", color='red', lw=1)
ax1.spines['left'].set_color('blue')
ax2.spines['right'].set_color('red')
ax1.tick_params(axis='y', colors='blue', which="both")
ax2.tick_params(axis='y', colors='red', which="both")
plt.legend(loc=(0.5, 0.2), fontsize=7)
plt.savefig(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\19-01-23_full\temp.png")
