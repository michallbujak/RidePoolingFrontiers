import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from collections import Counter

from Individual_pricing.matching import matching_function
from pricing_functions import *

_cr = 0.3
_num = 150
_sample = 25

with open("example_data_"+str(_num), "rb") as file:
    databanks_list, settings_list, params = pickle.load(file)

with open("results_" + str(_num) + "_" + str(_sample) + "_v4.pickle", "rb") as _file:
    data = pickle.load(_file)

rr = data["exmas"]["recalibrated_rides"]
# rr['02_accepted'] = rr.apply(check_prob_if_accepted, axis=1, discount=0.2)
# rr['02_revenue'] = rr.apply(lambda x: x["best_profit"][2]*x["02_accepted"], axis=1)
# rr['03_accepted'] = rr.apply(check_prob_if_accepted, axis=1, discount=0.3)
# rr['03_revenue'] = rr.apply(lambda x: x["best_profit"][2]*x["03_accepted"], axis=1)
rr['02_profit_02'] = rr.apply(lambda x: x["best_profit"][2]*x["02_accepted"] - x["u_veh"]*6*0.0015*0.2, axis=1)
rr['02_profit_04'] = rr.apply(lambda x: x["best_profit"][2]*x["02_accepted"] - x["u_veh"]*6*0.0015*0.4, axis=1)
rr['02_profit_06'] = rr.apply(lambda x: x["best_profit"][2]*x["02_accepted"] - x["u_veh"]*6*0.0015*0.6, axis=1)

data2 = matching_function(
    databank=data.copy(),
    # params={"matching_obj": ["02_revenue", "03_revenue"]},
    objectives=["02_profit_02", "02_profit_04", "02_profit_06"],
    min_max="max"
)
data["exmas"]["recalibrated_rides"] = data2["exmas"]["recalibrated_rides"]
for o in ["02_profit_02", "02_profit_04", "02_profit_06"]:
    data["exmas"]["schedules"][o] = data2["exmas"]["schedules"][o]

z = 0

# with open("results_" + str(_num) + "_" + str(_sample) + "_v4.pickle", "wb") as _file:
#     pickle.dump(data, _file)

