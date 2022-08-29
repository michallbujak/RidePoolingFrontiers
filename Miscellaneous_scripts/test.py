import pandas as pd
import numpy as np
import pickle
import scipy.stats as ss
import Topology.utils_topology as utils
from test2 import func

pp = utils.inverse_normal([1, 0], [1, 1])
print(pp(0.9, 0.9))


