import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from collections import Counter

from pricing_functions import *

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
shared = rr.loc[[len(t) > 1 for t in rr['indexes']]].copy()

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
            _df2[j].append(v[j - 1])

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

plot_discounts = False
if plot_discounts:
    discounts = shared["best_profit"].apply(lambda x: x[1])
    discounts = [a for b in discounts for a in b]
    discounts_revenue = shared.loc[shared['selected_expected_revenue'] == 1, "best_profit"].apply(lambda x: x[1])
    discounts_revenue = [a for b in discounts_revenue for a in b]
    discounts_profit20 = shared.loc[shared['selected_expected_profit_int_20'] == 1, "best_profit"].apply(lambda x: x[1])
    discounts_profit20 = [a for b in discounts_profit20 for a in b]
    discounts_profit40 = shared.loc[shared['selected_expected_profit_int_40'] == 1, "best_profit"].apply(lambda x: x[1])
    discounts_profit40 = [a for b in discounts_profit40 for a in b]
    discounts_profit60 = shared.loc[shared['selected_expected_profit_int_60'] == 1, "best_profit"].apply(lambda x: x[1])
    discounts_profit60 = [a for b in discounts_profit60 for a in b]
    discounts_no_select = shared.loc[(shared['selected_expected_revenue'] == 0)
                                 & (shared['selected_expected_profit_int_20'] == 0)
                                 & (shared['selected_expected_profit_int_40'] == 0)
                                 & (shared['selected_expected_profit_int_60'] == 0), "best_profit"].apply(lambda x: x[1])
    discounts_no_select = [a for b in discounts_no_select for a in b]

    fig, ax = plt.subplots()
    sns.kdeplot(discounts, color='green', ax=ax, label="All")
    sns.kdeplot(discounts_revenue, color='lightcoral', ax=ax, label="Revenue")
    sns.kdeplot(discounts_profit20, color='indianred', ax=ax, label="Profit 20")
    sns.kdeplot(discounts_profit40, color='brown', ax=ax, label="Profit 40")
    sns.kdeplot(discounts_profit60, color='darkred', ax=ax, label="Profit 60")
    sns.kdeplot(discounts_no_select, color='blue', ax=ax, label="Not selected")

    ax.legend(bbox_to_anchor=(1.02, 1.02), loc='upper left')
    plt.tight_layout()
    plt.savefig('discount_density.png', dpi=200)

    d_list = [discounts, discounts_revenue, discounts_profit20, discounts_profit40, discounts_profit60]

    print(pd.DataFrame({'mean': [np.mean(t) for t in d_list]},
                       index=["discount", "discounts_revenue", "discounts_profit20", "discounts_profit40", "discounts_profit60"]))


prob_distribution = True
if prob_distribution:
    data = shared
    data["d_avg_prob"] = data.apply(check_prob_if_accepted, axis=1, discount=0.203369)
    objectives = ["selected",
                  "selected_expected_revenue",
                  "selected_expected_profit_int_20",
                  "selected_expected_profit_int_40",
                  "selected_expected_profit_int_60"]
    names = ["Flat",
             "Revenue",
             "Profit 20",
             "Profit 40",
             "Profit 60"]
    selected = {
        objective: (data.loc[rr[objective] == 1], name) for objective, name in zip(objectives, names)
    }

    fig, ax = plt.subplots()

    for num, (sel, name) in enumerate(selected.values()):
        if num == 0:
            dat = sel["d_avg_prob"]
        else:
            dat = sel["best_profit"].apply(lambda x: x[3])
        # sns.histplot(dat, color=list(mcolors.BASE_COLORS.keys())[num],
        #              cumulative=True, label=name, kde=True, alpha=0.2,
        #              stat="density", element="step")
                     # log_scale=True, element="step", fill=False,
                     # cumulative=True, stat="density", label=name)
        # sns.ecdfplot(dat, color=list(mcolors.BASE_COLORS.keys())[num], label=name)
        sns.kdeplot(dat, color=list(mcolors.BASE_COLORS.keys())[num], label=name, bw_adjust=2)
    ax.legend(bbox_to_anchor=(1.02, 1.02), loc='upper left')
    ax.set_xlim(0, 1)
    plt.xlabel("Acceptance probability")
    plt.tight_layout()
    plt.savefig("probability_shared.png", dpi=200)

    x = 0
