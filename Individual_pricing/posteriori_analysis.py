import pickle
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from collections import Counter
import matplotlib.pylab as pylab

from pricing_functions import *
from matching import matching_function
from pricing_utils.batch_preparation import get_parameters

_cr = 0.3
_num = 100
_sample = 10

performance = True
plot_degrees = True
plot_discounts = True
kde_plot = False
prob_distribution = True
res_analysis = True

with open(r"C:\Users\zmich\Documents\GitHub\ExMAS_sideline\Individual_pricing\data\test\results_["
          + str(_num) + ", " + str(_num) + "]_" + str(_sample) + ".pickle",
          "rb") as file:
    data = pickle.load(file)[0]
# with open("results_" + str(_num) + "_" + str(_sample) + "_v4.pickle", "rb") as _file:
#     data = pickle.load(_file)

os.chdir(os.path.join(os.getcwd(), "data/figs"))

rr = data["exmas"]["recalibrated_rides"]
singles = rr.loc[[len(t) == 1 for t in rr['indexes']]].copy()
shared = rr.loc[[len(t) > 1 for t in rr['indexes']]].copy()

names_discs = ['012', '02']
discs = [0.12, 0.2]

for name, disc in zip(names_discs, discs):
    rr[name + '_accepted'] = rr.apply(lambda x: all([calculate_delta_utility(
        discount=disc,
        price=1.5 / 1000,
        ns_trip_dist=x['individual_distances'][j],
        vot=0.0046,
        wts=1.14756,
        travel_time_ns=x['individual_distances'][j] / 6,
        travel_time_s=x['individual_times'][j]
    ) > 0 for j in range(len(x['indexes']))]),
                                      axis=1)
    data = matching_function(
        databank=data,
        objectives=['u_veh'],
        min_max='min',
        filter_rides=name + '_accepted',
        opt_flag=name
    )
    rr["avg_prob_" + name] = rr.apply(check_prob_if_accepted, axis=1, discount=disc)

if performance:
    for obj in data['exmas']['objectives']:
        obj_no_int = obj.replace('_int', '')
        print(f"RIDE-HAILING: {obj}:\n {sum(singles[obj_no_int])} ")
        print(f"RIDE-POOLING: {obj}:\n {sum(data['exmas']['schedules'][obj][obj_no_int])} \n")

if plot_degrees:
    objectives_to_plot = ['profitability'] + ['expected_profit_int_' + str(t) for t in [20, 40, 60]]
    objectives_labels = ['Profitability', 'OC02', 'OC04', 'OC06']

    _d = {}
    for obj in data['exmas']['objectives']:
        _d[obj] = [len(t) for t in data['exmas']['schedules'][obj]["indexes"]]

    max_deg = max(max(j for j in tt) for tt in _d.values())

    _df = {}
    for k, v in _d.items():
        c = Counter(v)
        _df[k] = [c[j] for j in range(1, max(v) + 1)]

    _df = {k: v for k, v in _df.items() if
           k in objectives_to_plot}
    _df2 = {j: [] for j in range(1, max_deg + 1)}
    for k, v in _df.items():
        for j in range(1, max_deg + 1):
            _df2[j].append(v[j - 1] if j <= len(v) else 0)

    x = np.arange(len(_df.keys()))
    width = 0.2  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots()

    for attribute, measurement in _df2.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=0)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of rides')
    # ax.set_title('Degree distribution')
    # ax.set_xticks(x + width, data['exmas']['objectives'])
    ax.set_xticks(x + width, objectives_labels)
    lgd = ax.legend(title='Degree', bbox_to_anchor=(1.02, 1), borderaxespad=0, ncols=2, loc='upper left')
    ax.set_ylim(0, max(max(t) for t in _df.values()) + 2)
    # ax.set_xlabel("Expected profit with operating cost of")
    # plt.show()
    plt.savefig('degrees_' + str(_sample) + '.png', dpi=200,
                bbox_extra_artists=(lgd,), bbox_inches='tight')

if plot_discounts:
    objectives = ['profitability'] + ['expected_profit_int_' + str(t) for t in [20, 40, 60]]
    objectives_names = ['Profitability'] + ["Expected Profit OC" + str(t) for t in [20, 40, 60]]
    discounts = {'all': shared["best_profit"].apply(lambda x: x[1])}
    discounts['all'] = [a for b in discounts['all'] for a in b]
    for objective in objectives:
        discounts[objective] = shared.loc[shared['selected_' + objective] == 1, "best_profit"].apply(lambda x: x[1])
        discounts[objective] = [a for b in discounts[objective] for a in b]

    discounts['no_select'] = shared.loc[pd.concat([shared['selected_' + obj] == 1 for obj in objectives],
                                                  axis=1).apply(lambda x: not any(t for t in x),
                                                                axis=1), 'best_profit'].apply(lambda x: x[1])
    discounts['no_select'] = [a for b in discounts['no_select'] for a in b]

    colors = list(mcolors.BASE_COLORS)
    labels = ['All', 'Profit'] + ['OC0' + str(j) for j in [2, 4, 6]]
    fig, ax = plt.subplots()
    if kde_plot:
        for num, obj in enumerate(['all'] + objectives):
            sns.histplot(discounts[obj], ax=ax, label=labels[num], color=colors[num])


        def upper_rugplot(data, height=.02, _ax=None, **kwargs):
            from matplotlib.collections import LineCollection
            _ax = _ax or plt.gca()
            kwargs.setdefault("linewidth", 0.1)
            kwargs.setdefault("color", "green")
            segs = np.stack((np.c_[data, data],
                             np.c_[np.ones_like(data), np.ones_like(data) - height]),
                            axis=-1)
            lc = LineCollection(segs, transform=_ax.get_xaxis_transform(), **kwargs)
            _ax.add_collection(lc)


        upper_rugplot(discounts['all'], _ax=ax, color=colors[0])
        sns.rugplot(discounts['profitability'], color=colors[1])
        plt.axvline(0.05, label='Guaranteed discount', ls=':', lw=0.5, color='black')
        # sns.kdeplot(discounts_no_select, color='blue', ax=ax, label="Not selected")

        ax.legend(bbox_to_anchor=(1.02, 1.02), loc='upper left')
        plt.tight_layout()
        plt.savefig('discount_density_' + str(_sample) + '_rug.png', dpi=200)

    else:
        obj_discounts = {k: v for k, v in discounts.items() if k in objectives}
        no_obj_discounts = {k: v for k, v in discounts.items() if k not in objectives}
        plt.hist(list(obj_discounts.values()), stacked=False, density=True, label=objectives_names,
                 weights=[[1 / max(t)] * len(t) for t in list(obj_discounts.values())])
        # plt.axvline(0.05, label='Guaranteed discount', ls=':', lw=0.5, color='black')
        plt.yticks([])
        ax.legend(bbox_to_anchor=(1.02, 1.02), loc='upper left')
        plt.tight_layout()
        plt.savefig('discounts_density_objectives_' + str(_sample) + '.png', dpi=200)

    d_list = discounts.values()
    mean_profit_disc = np.mean(list(d_list)[1])
    print(pd.DataFrame({'mean': [np.mean(t) for t in d_list]},
                       index=list(discounts.keys())))

if prob_distribution or res_analysis:
    shared = rr.loc[[len(t) > 1 for t in rr['indexes']]].copy()
    shared["prob"] = shared["best_profit"].apply(lambda x: x[3])
    rr = data["exmas"]["recalibrated_rides"]
    rr["prob"] = rr["best_profit"].apply(lambda x: x[3])

    objectives = ["selected_u_veh" + names_discs[0],
                  "selected_u_veh" + names_discs[1],
                  "selected_profitability",
                  "selected_expected_profit_int_20",
                  "selected_expected_profit_int_40",
                  "selected_expected_profit_int_60"]
    names = ["Flat disc. 0." + str(discs[0])[1:] + " Revenue",
             "Flat disc. 0." + str(discs[1])[1:] + " Revenue",
             "Pers. Profitability",
             "Pers. Profit OC 0.2",
             "Pers. Profit OC 0.4",
             "Pers. Profit OC 0.6"]
    selected = {
        objective: (shared.loc[rr[objective] == 1], name) for objective, name in zip(objectives, names)
    }

if prob_distribution:
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (8, 5),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    pylab.rcParams.update(params)

    fig, ax = plt.subplots()

    for num, (sel, name) in enumerate(selected.values()):
        if num == 0:
            dat = sel["avg_prob_" + names_discs[0]]
        elif num == 1:
            dat = sel["avg_prob_" + names_discs[1]]
        else:
            dat = sel["prob"].apply(np.prod)
        # sns.histplot(dat, color=list(mcolors.BASE_COLORS.keys())[num],
        #              cumulative=False, label=name, kde=False, alpha=0.1,
        #              stat="frequency", element="step")
        # log_scale=True, element="step", fill=False,
        # cumulative=True, stat="density", label=name)
        # sns.ecdfplot(dat, color=list(mcolors.BASE_COLORS.keys())[num], label=name)
        sns.kdeplot(dat, color=list(mcolors.BASE_COLORS.keys())[num], label=name, bw_adjust=1)
    # ax.legend(bbox_to_anchor=(1.02, 1.02), loc='upper left')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)
    # plt.xlabel("Acceptance probability")
    plt.xlabel(None)
    plt.tight_layout()
    # plt.savefig("probability_shared_" + str(_sample) + ".png", dpi=200)
    plt.savefig("probability_shared_" + str(_sample) + "_sel.png", dpi=200)

    rr['prob'] = rr['prob'].apply(np.prod)

    fig, ax = plt.subplots()

    for obj, lab in [("avg_prob_" + names_discs[0], "Flat discount 0." + names_discs[0][1:]),
                     ("avg_prob_" + names_discs[1], "Flat discount 0." + names_discs[1][1:]),
                     ("prob", "Personalised")]:
        r_s = rr.loc[[len(t) != 1 for t in rr["indexes"]]]
        # sns.histplot(data[obj], cumulative=False, label=lab, kde=False, alpha=0.1,
        #              stat="frequency", element="step")
        sns.kdeplot(r_s[obj], label=lab, bw_adjust=1)
    # ax.legend(bbox_to_anchor=(1.02, 1.02), loc='upper left')
    ax.legend(loc='upper left')
    # plt.ylabel("Density", fontsize=13)
    ax.set_xlim(0, 1)
    plt.xlabel(None)
    # plt.xlabel("Acceptance probability")
    plt.tight_layout()
    plt.savefig("probability_shared_" + str(_sample) + "_all.png", dpi=200)

    plt.close()
    fig, ax = plt.subplots()
    for _n, obj, lab in [
        ('selected_u_veh' + names_discs[0], "avg_prob_" + names_discs[0], "Flat discount 0." + names_discs[0][1:]),
        ('selected_u_veh' + names_discs[1], "avg_prob_" + names_discs[1], "Flat discount 0." + names_discs[1][1:]),
        ('selected_profitability', "prob", "Personalised")]:
        r_s = rr.loc[[len(t) != 1 for t in rr["indexes"]]]
        r_s = r_s.loc[[bool(t) for t in r_s[_n]]]
        plt.hist(list(r_s[obj]), label=lab)
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)
    plt.xlabel(None)
    plt.tight_layout()
    plt.savefig("probability_shared_" + str(_sample) + "_sel_v2.png", dpi=200)

if res_analysis:
    schedules = data['exmas']['schedules']

    for name, schedule in schedules.items():
        if name[:2] == "ex" or name=='profitability':
            schedule["prob"] = schedule["best_profit"].apply(lambda x: x[3])
        else:
            col = name[5:] + "_accepted"
            schedule["prob"] = schedule[col]

        schedule["dist_saved"] = schedule['ttrav_ns'] - schedule['ttrav']
        schedule["e_dist_saved"] = schedule.apply(lambda x: x["prob"] * x["dist_saved"], axis=1)

    # measures = ['u_veh', 'revenue', 'expected_revenue', 'expected_profit_20',
    #             'expected_profit_30', 'expected_profit_40', 'expected_profit_50',
    #             'expected_profit_60', 'dist_saved', 'e_dist_saved']
    measures = ['profitability', 'expected_profit_20',
                'expected_profit_40', 'expected_profit_60', 'e_dist_saved']

    results = {}
    for meas in measures:
        results[meas] = [sum(t[meas]) for t in schedules.values()]

    # results["dist_saved"] = [6*sum(t['ttrav_ns'] - t['ttrav']) for t in schedules.values()]
    # results["dist_veh"] = [6*t for t in results["u_veh"]]
    # del results["u_veh"]
    results = pd.DataFrame(results)
    results.index = schedules.keys()
    results = results.round()
    # results = results.drop(columns=["expected_profit_30", "expected_profit_50"])
    # results = results.drop(labels=["expected_profit_int_30", "expected_profit_int_50"])
    print(results.to_latex())
