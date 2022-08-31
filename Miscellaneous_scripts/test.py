import pandas as pd
import numpy as np
import pickle
import scipy.stats as ss
import Topology.utils_topology as utils
from test2 import func

with open("data/31-08-22/final_res_31-08-22.obj", "rb") as file:
    x = list(pickle.load(file))

z = 0
