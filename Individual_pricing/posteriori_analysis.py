import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

_cr = 0.3
_num = 150
_sample = 10

# with open("example_data_"+str(_num), "rb") as file:
#     databanks_list, settings_list, params = pickle.load(file)

# with open("results_" + str(_num) + "_" + str(_sample) + "_0" + str(int(10 * _cr)) + "_v2.pickle", "rb") as _file:
#     data = pickle.load(_file)[0]

with open("results_" + str(_num) + "_" + str(_sample) + "_v3.pickle", "rb") as _file:
    data = pickle.load(_file)[0]

rr = data["exmas"]["recalibrated_rides"]
singles = rr.loc[[len(t) == 1 for t in rr['indexes']]].copy()

for obj in data['exmas']['objectives']:
    obj_no_int = obj.replace('_int', '')
    print(f"RIDE-HAILING: {obj}:\n {sum(singles[obj_no_int])} ")
    print(f"RIDE-POOLING: {obj}:\n {sum(data['exmas']['schedules'][obj][obj_no_int])} \n")

plot = False
if plot:
    _d = {}
    for obj in data['exmas']['objectives']:
        _d[obj] = [len(t) for t in data['exmas']['schedules'][obj]["indexes"]]

    _df = {}
    for k, v in _d.items():
        c = Counter(v)
        _df[k] = [c[j] for j in range(1, 4)]

    _df = {k: v for k, v in _df.items() if
           k in ['expected_revenue',
                 'expected_profit_int_20',
                 'expected_profit_int_40',
                 'expected_profit_int_60']}
    _df2 = {j: [] for j in range(1, 4)}
    for k, v in _df.items():
        for j in range(1, 4):
            _df2[j].append(v[j-1])

    x = np.arange(len(_df.keys()))
    width = 0.25  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots()

    for attribute, measurement in _df2.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=0)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of rides')
    # ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width, ['Revenue', 'Profit 20', 'Profit 40', 'Profit 60'])
    ax.legend(title='Degree', loc='upper left', ncols=3)
    ax.set_ylim(0, 80)

    plt.savefig('degrees.png', dpi=200)

x = 0