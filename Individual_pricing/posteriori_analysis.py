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
_sample = 25

plot_degrees = False
plot_discounts = False
prob_distribution = False
res_analysis = True


with open("results_" + str(_num) + "_" + str(_sample) + "_v4.pickle", "rb") as _file:
    data = pickle.load(_file)

rr = data["exmas"]["recalibrated_rides"]
singles = rr.loc[[len(t) == 1 for t in rr['indexes']]].copy()
shared = rr.loc[[len(t) > 1 for t in rr['indexes']]].copy()

for obj in data['exmas']['objectives']:
    obj_no_int = obj.replace('_int', '')
    print(f"RIDE-HAILING: {obj}:\n {sum(singles[obj_no_int])} ")
    print(f"RIDE-POOLING: {obj}:\n {sum(data['exmas']['schedules'][obj][obj_no_int])} \n")

if plot_degrees:
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

    plt.savefig('degrees_' + str(_sample) + '.png', dpi=200)

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
    plt.savefig('discount_density_' + str(_sample) + '.png', dpi=200)

    d_list = [discounts, discounts_revenue, discounts_profit20, discounts_profit40, discounts_profit60]

    print(pd.DataFrame({'mean': [np.mean(t) for t in d_list]},
                       index=["discount", "discounts_revenue", "discounts_profit20", "discounts_profit40", "discounts_profit60"]))


if prob_distribution or res_analysis:
    dat = shared
    dat["prob"] = dat["best_profit"].apply(lambda x: x[3])
    # data["d_avg_prob"] = data.apply(check_prob_if_accepted, axis=1, discount=0.203369)
    objectives = ["selected_02_revenue",
                  "selected_03_revenue",
                  "selected_expected_revenue",
                  "selected_expected_profit_int_20",
                  "selected_expected_profit_int_40",
                  "selected_expected_profit_int_60"]
    names = ["Flat disc. 0.2 Revenue",
             "Flat disc. 0.3 Revenue",
             "Pers. Revenue",
             "Pers. Profit OC 0.2",
             "Pers. Profit OC 0.4",
             "Pers. Profit OC 0.6"]
    selected = {
        objective: (dat.loc[rr[objective] == 1], name) for objective, name in zip(objectives, names)
    }

if prob_distribution:
    fig, ax = plt.subplots()

    for num, (sel, name) in enumerate(selected.values()):
        if num == 0:
            dat = sel["02_accepted"]
        elif num == 1:
            dat = sel["03_accepted"]
        else:
            dat = sel["prob"]
        sns.histplot(dat, color=list(mcolors.BASE_COLORS.keys())[num],
                     cumulative=False, label=name, kde=False, alpha=0.1,
                     stat="frequency", element="step")
                     # log_scale=True, element="step", fill=False,
                     # cumulative=True, stat="density", label=name)
        # sns.ecdfplot(dat, color=list(mcolors.BASE_COLORS.keys())[num], label=name)
        # sns.kdeplot(dat, color=list(mcolors.BASE_COLORS.keys())[num], label=name, bw_adjust=1)
    ax.legend(bbox_to_anchor=(1.02, 1.02), loc='upper left')
    ax.set_xlim(0, 1)
    plt.xlabel("Acceptance probability")
    plt.tight_layout()
    # plt.savefig("probability_shared_" + str(_sample) + ".png", dpi=200)
    plt.savefig("probability_shared_" + str(_sample) + "_hist.png", dpi=200)

    fig, ax = plt.subplots()

    for obj, lab in [("02_accepted", "Flat disc. 0.2"),
                     ("03_accepted", "Flat disc. 0.3"),
                     ("prob", "Personalised")]:
        # sns.histplot(data[obj], cumulative=False, label=lab, kde=False, alpha=0.1,
        #              stat="frequency", element="step")
        sns.kdeplot(data[obj], label=lab, bw_adjust=1)
    ax.legend(bbox_to_anchor=(1.02, 1.02), loc='upper left')
    ax.set_xlim(0, 1)
    plt.xlabel("Acceptance probability")
    plt.tight_layout()
    plt.savefig("probability_shared_all_" + str(_sample) + "_hist.png", dpi=200)


if res_analysis:
    schedules = data['exmas']['schedules']
    measures = ['u_veh', 'revenue', 'expected_revenue', 'expected_profit_20',
                'expected_profit_30', 'expected_profit_40', 'expected_profit_50',
                'expected_profit_60']
    results = {}
    for meas in measures:
        results[meas] = [sum(t[meas]) for t in schedules.values()]

    results["dist_saved"] = [6*sum(t['ttrav_ns'] - t['ttrav']) for t in schedules.values()]
    results["dist_veh"] = [6*t for t in results["u_veh"]]
    del results["u_veh"]
    results = pd.DataFrame(results)
    results.index = schedules.keys()
    results = results.round()
    results = results.drop(columns=["expected_profit_30", "expected_profit_50"])
    results = results.drop(labels=["expected_profit_int_30", "expected_profit_int_50"])
    print(results.to_latex())
