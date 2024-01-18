import pandas as pd
import osmnx as ox
import networkx as nx
from dotmap import DotMap
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pickle
import dotmap
from Utils import utils_topology as utils
import os
import ExMAS
from tqdm import tqdm
import logging
import time
import itertools


os.chdir("data/res")

results = {}
for num, frac, rep in list(itertools.product([1, 2], [0.001, 0.003], range(6))):
    results[str((num, frac, rep))] = pd.read_csv('KPI_hub' + str(num) + "_" + str(frac) + "_" + str(rep) + ".csv")["KPI"]

indexes = pd.read_csv('KPI_hub' + str(num) + "_" + str(frac) + "_" + str(rep) + ".csv", index_col=0).index
df = pd.DataFrame(results)
df.index = indexes
print(df)

