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


with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Miscellaneous_scripts\data\22-07-24\150_old_22-07-24.obj",'rb') as file:
    e = pickle.load(file)

ut.plot_demand(e, dpi=400, origin_colour="red", destination_colour="blue")
