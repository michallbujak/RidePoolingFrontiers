import pickle
import os
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import PercentFormatter
from matplotlib.collections import PathCollection
import seaborn as sns

from pricing_functions import _expected_flat_measures
from pricing_functions import *
from matching import matching_function
from auxiliary_functions import extract_selected_discounts, extract_selected_profitability, bracket


parser = argparse.ArgumentParser()
parser.add_argument("--no-requests", type=int, required=True)
parser.add_argument("--sample-size", type=int, required=True)
parser.add_argument("--legacy-version", action="store_true")
parser.add_argument("--baselines", action="store_false")
parser.add_argument("--profitability", action="store_false")
parser.add_argument("--min-accept", type=float, default=0.1)
parser.add_argument("--operating-cost", type=float, default=0.5)
parser.add_argument("--analysis-parts", nargs='+', type=int, default=[1] * 7)
parser.add_argument("--discounts", nargs='+', type=int, default=[0.2])
parser.add_argument("--avg-speed", type=float, default=6)
parser.add_argument("--price", type=float, default=1.5)
parser.add_argument("--guaranteed-discount", type=float, default=0.05)
parser.add_argument("--separate", action='store_false')
parser.add_argument("--dpi", type=int, default=300)
parser.add_argument("--pic-format", type=str, default='png')
args = parser.parse_args()
print(args)

_num = args.no_requests
_sample = args.sample_size

if os.getcwd()[-len("Individual_pricing"):] == "Individual_pricing":
    path = os.getcwd()
else:
    path = path_joiner(os.getcwd(), "Individual_pricing")
path = path_joiner(path, 'data')
path = path_joiner(path, str(_num) + '_' + str(_sample))

if args.legacy_version:
    if args.profitability:
        path = path_joiner(path, 'results_[' + str(_num) + ', ' + str(_num) + ']_' + str(_sample))
    else:
        res_name = 'results_[' + str(_num) + ', ' + str(_num) + ']_' + str(_sample)
        res_name += "_" + str(args.operating_cost) + "_" + str(args.min_accept)
        path = path_joiner(path, res_name)
else:
    path = path_joiner(path, 'results_' + str(_num) + '_' + str(_sample))


if args.legacy_version:
    try:
        with open(path + '_amended.pickle', "rb") as file:
            data = pd.read_pickle(file)
    except FileNotFoundError:
        with open(path + '.pickle', "rb") as file:
            data = pd.read_pickle(file)[0]
else:
    if args.baselines:
        path += '_baselines'
    if Path(path+'_amended.pickle').is_file():
        with open(path + '_amended.pickle', "rb") as file:
            data = pd.read_pickle(file)
    else:
        with open(path + '.pickle', "rb") as file:
            if args.baselines:
                data = pd.read_pickle(file)
            else:
                data = pd.read_pickle(file)[0]

os.chdir(os.path.join(os.getcwd(), "results"))
res_path = os.path.join(os.getcwd(), str(_num) + "_" + str(_sample))

try:
    os.chdir(res_path)
except FileNotFoundError:
    os.mkdir(res_path)
    os.chdir(res_path)

if not args.profitability:
    res_path = os.path.join(res_path,
                            str(args.operating_cost) + "_" + str(args.min_accept))
    try:
        os.chdir(res_path)
    except FileNotFoundError:
        os.mkdir(res_path)
        os.chdir(res_path)

if args.legacy_version:
    rr = data["exmas"]["recalibrated_rides"]
else:
    rr = data["exmas"]["rides"]
schedule_profit = data['exmas']['schedules']['profitability']
schedule_profit_sh = schedule_profit.loc[[len(t) > 1 for t in schedule_profit['indexes']]]

avg_discount = np.mean(
    [a for b in schedule_profit_sh['best_profit'].apply(lambda x: list(x[1])) for a in b]
)
avg_discount = round(avg_discount, 2)

discounts = [avg_discount] + args.discounts
discounts_names = ['0' + str(int(100 * disc)) for disc in discounts]
discounts_labels = ['Personalised'] + ['Flat ' + str(d) for d in discounts]

# Conduct analysis for flat discount if not conducted yet
if data.get('flat_analysis') == discounts:
    pass
else:
    for name, disc in zip(discounts_names, discounts):
        rr[name + '_accepted'] = rr.apply(lambda x: all([calculate_delta_utility(
            discount=disc,
            price=1.5 / 1000,
            ns_trip_dist=x['individual_distances'][j],
            vot=0.0046,
            wts=1.14756,
            travel_time_ns=x['individual_distances'][j] / args.avg_speed,
            travel_time_s=x['individual_times'][j]
        ) > 0 for j in range(len(x['indexes']))]),
                                          axis=1)

        rr["prod_prob_" + name] = rr.apply(check_prob_if_accepted, axis=1, discount=disc)
        rr["probs_" + name] = rr.apply(check_prob_if_accepted, axis=1, discount=disc, total=False)

        rr['rev_cost'] = rr.apply(
            lambda x: _expected_flat_measures(
                vector_probs=x["probs_" + name],
                shared_dist=x['u_veh'] * args.avg_speed,
                ind_dists=x['individual_distances'],
                price=args.price,
                sharing_disc=disc,
                guaranteed_disc=args.guaranteed_discount
            ),
            axis=1
        )

        rr[name + '_profitability'] = rr.apply(
            lambda row: len(row['indexes']) * row['rev_cost'][0] / row['rev_cost'][1]
            if row['rev_cost'][1] != 0 else 0,
            axis=1
        )

        for oc in discounts:
            rr[name + '_profit_' + str(int(100 * oc))] = rr['rev_cost'].apply(
                lambda el: el[0] - oc * el[1]
            ).copy()
            data = matching_function(
                databank=data,
                objectives=[name + '_profit_' + str(int(100 * oc))],
                min_max='max',
                filter_rides=False,  # name + '_accepted',
                opt_flag=""
            )

        data = matching_function(
            databank=data,
            objectives=[name + '_profitability'],
            min_max='max',
            filter_rides=False,  # name + '_accepted',
            opt_flag=""
        )
    data['flat_analysis'] = discounts

    with open(path + '_amended.pickle', "wb") as file:
        pickle.dump({
            'exmas': data['exmas'],
            'flat_analysis': data['flat_analysis']
        }, file)

    raise Warning("Please, rerun the project to ensure the data integrity")

# Now analysis
if args.baselines:
    for baseline_name in ['jiao', 'karaenke', 'detour']:
        rr['prod_prob_' + baseline_name] = rr['baseline_' + baseline_name].apply(lambda x: np.prod(x[3]))
        schedule_trio = data['exmas']['schedules'][baseline_name].apply(
            lambda y: (tuple(y['indexes']), y['u_pax'], y['u_veh']), axis=1)
        rr_trio = rr.apply(lambda x: (tuple(x['indexes']), x['u_pax'], x['u_veh']), axis=1)
        rr['selected_' + baseline_name] = [any(outer == inner for inner in schedule_trio) for outer in rr_trio]
        rr['ex_distance'] = rr['baseline_' + baseline_name].apply(lambda x: x[4])

singles = rr.loc[[len(t) == 1 for t in rr['indexes']]].copy()
shared = rr.loc[[len(t) > 1 for t in rr['indexes']]].copy()

shared["prob"] = shared["best_profit"].apply(lambda x: np.prod(x[3]))
rr["prob"] = rr["best_profit"].apply(lambda x: np.prod(x[3]))

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)

if args.analysis_parts[0]:
    for obj in data['exmas']['objectives']:
        obj_no_int = obj.replace('_int', '')
        print(f"RIDE-HAILING: {obj}:\n {sum(singles[obj_no_int])} ")
        print(f"RIDE-POOLING: {obj}:\n {sum(data['exmas']['schedules'][obj][obj_no_int])} \n")

if args.analysis_parts[1]:
    objectives_to_plot = ['profitability'] + [t + "_profitability" for t in discounts_names]
    if args.baselines:
        objectives_to_plot += ['jiao', 'karaenke', 'detour']
    _d = {}
    for obj in objectives_to_plot:
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

    if args.baselines:
        _labels = discounts_labels + ['Jiao', 'Karaenke', 'Detour']
    else:
        _labels = discounts_labels

    ax.set_ylabel('Number of rides')
    ax.set_xticks(x + width, _labels)
    lgd = ax.legend(title='Degree', bbox_to_anchor=(1.02, 1), borderaxespad=0, ncols=2, loc='upper left')
    ax.set_ylim(0, max(max(t) for t in _df.values()) + 5)
    plt.savefig('degrees_' + str(_sample) + '.' + args.pic_format, dpi=args.dpi,
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    _df3 = pd.DataFrame(_df2)
    _df3['kind'] = _labels
    for num in range(2, max_deg + 1):
        _df3[num] = _df3[num].apply(lambda x: num*x)
    _df4 = pd.DataFrame(columns=['degree', 'kind'])
    for out_lab, (lab, row) in enumerate(_df3.iterrows()):
        cur_kind = _labels[out_lab]
        for deg in range(1, max_deg + 1):
            for en in range(row[deg]):
                _df4 = pd.concat([_df4, pd.DataFrame({'degree': [str(deg)], 'kind': [cur_kind]})])
    _df4 = _df4.reset_index(drop=True)
    fig, ax = plt.subplots()
    graph = sns.histplot(x='kind', data=_df4, hue='degree', multiple='stack', shrink=0.9,
                 hue_order=[str(t) for t in range(max_deg, 0, -1)])
    respective_height = [[0] for j in range(len(_labels))]
    for num, rect in enumerate(graph.patches):
        count = rect.get_height()
        respective_height[num%len(_labels)] += [count]
        tmp = respective_height[num%len(_labels)]
        graph.text(rect.get_x()+rect.get_width()/2-0.05, sum(tmp[:-1]) + tmp[-1]/2-2.5, str(int(count)),
                   color='white')
        if num >= len(graph.patches) - len(_labels):
            temp_val = sum(_deg*tmp[_deg] for _deg in range(1, max_deg+1))
            graph.text(rect.get_x() + rect.get_width() / 2 - 0.4, sum(tmp)+0.5,
                       'avg. ' + str(round(temp_val/150, 2)),
                       color='black')

    lgd = ax.legend(title='Degree', labels=range(1, 5), title_fontsize='x-large',
                    bbox_to_anchor=(1.15, 1), borderaxespad=0, loc='upper right')
    plt.xlabel(None)
    plt.ylabel(None)
    plt.yticks(np.arange(0, args.no_requests + 30, 30))
    ax.tick_params(axis='x', which='major', labelsize=10)
    plt.savefig('degrees_stacked_' + str(_sample) + '.' + args.pic_format, dpi=args.dpi,
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

if args.analysis_parts[2]:
    obj_discounts = {'all': [a for b in shared["best_profit"].apply(lambda x: x[1]) for a in b],
                     'selected': [a for b in
                                  shared.loc[shared['selected_profitability'] == 1,
                                  'best_profit'].apply(lambda x: x[1])
                                  for a in b]
                     }
    fig, ax = plt.subplots()
    plt.hist(list(obj_discounts.values()), stacked=False, density=True, label=['Shareability set', 'Offer'],
             weights=[[1 / max(t)] * len(t) for t in list(obj_discounts.values())],
             bins=np.arange(0, 0.55, 0.05).tolist())
    plt.xticks([round(t, 2) for t in np.arange(0.05, 0.55, 0.05)])
    plt.xlabel('Discount level')
    plt.ylabel('Density')
    ax.legend()
    ax.set_xlim(0.05, 0.55)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('discounts_density_' + str(_sample) + '.' + args.pic_format, dpi=args.dpi)
    plt.close()

    fig, ax = plt.subplots()
    sns.kdeplot(list(obj_discounts.values())[0], label='All')
    sns.kdeplot(list(obj_discounts.values())[1], label='Selected')
    ax.legend()
    ax.set_xlim(0.05, 0.55)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('discounts_density_kde_' + str(_sample) + '.' + args.pic_format, dpi=args.dpi)
    plt.close()

if args.analysis_parts[3] + args.analysis_parts[4] >= 1:
    selected_objectives = ['selected_profitability']
    selected_objectives += ['selected_' + t + '_profitability' for t in discounts_names]
    if args.baselines:
        selected_objectives += ['selected_' + t for t in ['jiao', 'karaenke', 'detour']]
        _labels = discounts_labels  + ['Jiao', 'Karaenke', 'Detour']
    else:
        _labels = discounts_labels
    selected = {
        objective: (shared.loc[rr[objective] == 1], name)
        for objective, name in zip(selected_objectives, _labels)
    }

if args.analysis_parts[3]:
    if args.baselines:
        col_labels = ["prob"] + ["prod_prob_" + t for t in discounts_names+['jiao', 'karaenke', 'detour']]
    else:
        col_labels = ["prob"] + ["prod_prob_" + t for t in discounts_names]
    dat = []
    labels = []
    for num, (sel, name) in enumerate(selected.values()):
        dat += [list(sel[col_labels[num]])]
        labels += [name]

    fig, ax = plt.subplots()
    plt.hist(dat, label=labels)
    ax.legend(loc='upper left')
    plt.xlabel(None)
    plt.tight_layout()
    plt.savefig("probability_shared_" + str(_sample) + "_sel." + args.pic_format, dpi=args.dpi)
    plt.close()

    fig, ax = plt.subplots()
    for _d, _l in zip(dat, labels):
        sns.kdeplot(_d, label=_l, bw_method=0.3, lw=2)
    # lgd = ax.legend(bbox_to_anchor=(1.02, 1), ncols=1, loc='upper left')
    plt.ylabel(None)
    ax.set_yticks([])
    plt.xlim(0, 1)
    plt.yticks(None)
    plt.xlabel('Acceptance probability', fontsize=15)
    plt.tight_layout()
    plt.savefig("probability_shared_" + str(_sample) + "_sel_kde." + args.pic_format, dpi=args.dpi)
    plt.close()

    dat = []
    labels = []
    for obj, label in zip(col_labels, _labels):
        r_s = rr.loc[[len(t) != 1 for t in rr["indexes"]]]
        dat += [list(r_s[obj])]
        labels += [label]

    fig, ax = plt.subplots()
    plt.hist(dat, label=labels)
    # ax.legend(loc='upper right')
    # ax.get_legend().remove()
    plt.xlabel(None)
    plt.tight_layout()
    plt.savefig("probability_shared_" + str(_sample) + "_all." + args.pic_format, dpi=args.dpi)
    plt.close()

    fig, ax = plt.subplots()
    for _d, _l in zip(dat, labels):
        sns.kdeplot(_d, label=_l, bw_method=0.2, lw=2)
    ax.legend(loc='upper right', fontsize=15)
    plt.xlim(0, 1)
    plt.ylabel('Density', fontsize=20)
    plt.yticks(None)
    ax.set_yticks([])
    plt.xlabel('Acceptance probability', fontsize=15)
    plt.tight_layout()
    plt.savefig("probability_shared_" + str(_sample) + "_all_kde." + args.pic_format, dpi=args.dpi)
    plt.close()

if args.analysis_parts[4]:
    results, results_list = extract_selected_discounts(rr, discounts_names)

    fig, ax = plt.subplots()

    for k, v in results.items():
        sns.kdeplot(v, label=k, bw_adjust=1)

    ax.legend(loc='upper left')
    plt.xlabel(None)
    plt.ylabel(None)
    plt.tight_layout()
    plt.savefig("profitability_" + str(_sample) + "_sel." + args.pic_format, dpi=args.dpi)
    plt.close()

    fig, ax = plt.subplots()
    plt.hist(results_list, label=['Personalised'] + discounts_names,
             stacked=False, density=True,
             weights=[[1 / max(t)] * len(t) for t in results_list])
    ax.legend(loc='upper right')
    plt.xlabel(None)
    plt.ylabel(None)
    plt.tight_layout()
    plt.savefig("profitability_" + str(_sample) + "_sel_hist." + args.pic_format, dpi=args.dpi)
    plt.close()

if args.analysis_parts[5]:
    profitability_scatter = [(x, num) for num, x in enumerate(list(shared['profitability']))]
    profitability_scatter.sort(reverse=False)
    profitability_scatter, permutation = zip(*profitability_scatter)
    profitability_scatter = [list(profitability_scatter)]
    for disc_name in discounts_names:
        dat = list(shared[disc_name + '_profitability'])
        dat = [dat[t] for t in permutation]
        profitability_scatter += [dat]
    degrees = [len(t) for t in shared['indexes']]
    degrees = list(Counter(degrees).values())
    degrees_positions = list(np.cumsum(degrees))
    degrees_positions = [int(t1 + t2 / 2)
                         for t1, t2 in zip([0] + degrees_positions[:-1], degrees)]

    fig, ax = plt.subplots()
    for num, cur_label in enumerate(discounts_labels):
        dat = profitability_scatter[num]
        if num == 0:
            size = 2
        else:
            size = 1
        plt.scatter(x=range(len(dat)), y=dat, label=discounts_labels[num], s=2,
                    edgecolors='none')

    for num, (deg_len, deg_post) in enumerate(zip(degrees, [0] + list(np.cumsum(degrees)))):
        bracket(ax, text=str(num + 2), pos=[deg_post, -0.01], scalex=deg_len,
                linekw={'color': "black", 'lw': 1}, textkw={'size': 12})

    # ax.set_xticks(degrees_positions, [t + 2 for t in range(len(degrees_positions))])
    ax.set_xticks([])
    lgnd = plt.legend(loc='upper left', fontsize=10)
    for handle in lgnd.legend_handles:
        handle.set_sizes([30])
    plt.savefig("scatter_all_profitability_balanced_" + str(_sample) + "." + args.pic_format, dpi=args.dpi)
    plt.close()

    shared['profitability_unbalanced'] = shared.apply(
        lambda row: row['profitability'] / len(row['indexes']),
        axis=1
    )
    for disc_name in discounts_names:
        shared[disc_name + '_profitability_unbalanced'] = shared.apply(
            lambda row: row[disc_name + '_profitability'] / len(row['indexes']),
            axis=1
        )
    if args.baselines:
        for baseline_name in ['jiao', 'karaenke', 'detour']:
            shared[baseline_name + '_profitability_unbalanced'] = shared.apply(
                lambda row: row['profitability_' + baseline_name + '_actual'] / len(row['indexes']),
                axis=1
            )

    shared_reordered = shared.copy()
    shared_reordered = shared_reordered.reset_index(drop=True)
    unbalanced = [shared_reordered.apply(
        lambda row: (row.name, 'Personalised', len(row['indexes']), row['profitability_unbalanced']),
        axis=1
    ).to_list()]
    unbalanced[0].sort(key=lambda it: (it[2], it[3]))
    new_order = [t[0] for t in unbalanced[0]]
    shared_reordered = shared_reordered.reindex(new_order)

    if args.baselines:
        iterable_labels = zip(discounts_names + ['jiao', 'karaenke', 'detour'],
                              discounts_labels[1:] + ['Jiao', 'Karaenke', 'Detour'])
    else:
        iterable_labels = zip(discounts_names, discounts_labels[1:])

    for _name, label in iterable_labels:
        unbalanced += [shared_reordered.apply(
            lambda row: (row.name, label, len(row['indexes']), row[_name + '_profitability_unbalanced']),
            axis=1
        ).to_list()]

    if args.separate:
        _range = range(2, 5)
    else:
        _range = [0]

    for _deg in _range:
        if args.separate:
            temp_data = [[ins for ins in outs if ins[2] == _deg] for outs in unbalanced]
        else:
            temp_data = unbalanced
        fig, ax = plt.subplots(figsize=(8,14))
        if args.baselines:
            iterable_labels = enumerate(discounts_labels + ['Jiao', 'Karaenke', 'Detour'])
        else:
            iterable_labels = enumerate(discounts_labels)
        for num, cur_label in iterable_labels:
            if num == 0:
                size = 12
            else:
                size = 10
            plt.scatter(x=range(len(temp_data[num])), y=[t[3] for t in temp_data[num]], label=cur_label, s=size,
                        edgecolors='none')

        if not args.separate:
            for num, (deg_len, deg_post) in enumerate(zip(degrees, [0] + list(np.cumsum(degrees)))):
                bracket(ax, text=str(num + 2), pos=[deg_post, -0.01], scalex=deg_len,
                        linekw={'color': "black", 'lw': 1}, textkw={'size': 12})

        ax.set_xticks([])
        plt.ylim(0.6, 2.8)
        plt.axhline(1.425, lw=2, ls='solid', color='red', label='Private ride')
        plt.xlabel('Individual shared rides', fontsize=20)
        if _deg != list(_range)[0]:
            plt.yticks([])
        else:
            plt.yticks(fontsize=25)
            plt.ylabel('Expected profitability', fontsize=25)
        if _deg == list(_range)[-1]:
            lgnd = plt.legend(loc='upper left', fontsize=25)
            for handle in lgnd.legend_handles:
                if type(handle) == PathCollection:
                    handle.set_sizes([200])
        plt.tight_layout()
        if args.separate:
            plt.savefig("scatter_all_profitability_unbalanced_" + str(_deg) + "_"
                        + str(_sample) + "." + args.pic_format, dpi=args.dpi)
        else:
            plt.savefig("scatter_all_profitability_unbalanced_" + str(_sample) + "." + args.pic_format, dpi=args.dpi)
        plt.close()

    shared_reordered['dist_prop'] = shared_reordered.apply(
        lambda row: 1 - row['u_veh'] * args.avg_speed / sum(row['individual_distances']),
        axis=1
    )

    output_list = shared_reordered.apply(
        lambda row: [row['dist_prop'], row['profitability_unbalanced']] +
                    [row[t + '_profitability_unbalanced'] for t in discounts_names],
        axis=1
    ).to_list()

    output_list.sort(key=lambda it: it[0])

    fig, ax = plt.subplots()
    for _num in range(len(output_list[0]) - 1):
        _size = 1 if _num != 0 else 2
        _lab = discounts_labels[_num]
        plt.scatter(x=[t[0] for t in output_list], y=[t[1 + _num] for t in output_list],
                    s=_size, label=_lab)
    lgnd = plt.legend(loc='upper left', fontsize=10)
    for handle in lgnd.legend_handles:
        handle.set_sizes([30])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    plt.tight_layout()
    plt.savefig('scatter_all_distance_saved_profitability_' + str(_sample) + "." + args.pic_format, dpi=args.dpi)
    plt.close()

    x, y = [t[0] for t in output_list], [t[1] for t in output_list]
    from numpy.polynomial.polynomial import Polynomial
    # _Poly = Polynomial.fit(x, y, 5)
    # x_plot, y_plot = _Poly.linspace(n=1000, domain=[min(x), max(x)])

    output_list_sel = shared_reordered.loc[shared_reordered['selected_profitability'] == 1]
    output_list_sel = output_list_sel.apply(
        lambda it: [it['dist_prop'], it['profitability_unbalanced']],
        axis=1
    )
    fig, ax = plt.subplots()
    plt.scatter(x=x, y=y, s=1, label='Shareability set', alpha=0.7)
    plt.scatter(x=[t[0] for t in output_list_sel], y=[t[1] for t in output_list_sel],
                s=5, label='Offer', color='red')
    # plt.plot(x_plot, y_plot, lw=1.5, color='red', label='Polynomial fit')
    plt.xlabel('Relative mileage reduction')
    plt.ylabel('Expected profitability')
    lgnd = plt.legend(loc='upper left', fontsize=15, markerscale=3)
    for handle in lgnd.legend_handles:
        handle.set_sizes([30])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    plt.tight_layout()
    plt.savefig('scatter_distance_saved_profitability_' + str(_sample) + "." + args.pic_format, dpi=args.dpi)

    shared_reordered['probs'] = shared_reordered['best_profit'].apply(lambda it: it[3])

    output_list = shared_reordered.apply(
        lambda row: [row['dist_prop'], row['probs']] +
                    [row['probs_' + t] for t in discounts_names],
        axis=1
    ).to_list()

    output_list2 = []
    for el in output_list:
        for j in range(len(el[1])):
            output_list2 += [[el[0]] + [el[t+1][j] for t in range(len(el)-1)]]

    output_list2.sort(key=lambda it: it[0])

    fig, ax = plt.subplots()
    for strategy_num, strategy_label in enumerate(discounts_labels):
        x, y = [t[0] for t in output_list2], [t[strategy_num+1] for t in output_list2]
        _Poly = Polynomial.fit(x, y, 5)
        x_plot, y_plot = _Poly.linspace(n=1000, domain=[min(x), max(x)])
        plt.plot(x_plot, y_plot, lw=1.5, label=strategy_label)
    plt.legend(loc='upper right', fontsize=10, markerscale=3)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    # plt.xlim(-0.45, 0.55)
    plt.xlabel("Mileage reduction")
    plt.ylabel("Acceptance probability")
    plt.tight_layout()
    plt.savefig('poly_distance_acceptance_' + str(_sample) + "." + args.pic_format, dpi=args.dpi)

    for _num, (disc_name, disc_lab) in enumerate(zip([""] + ["_" + t for t in discounts_names], discounts_labels)):
        _data_sel = shared_reordered.loc[shared_reordered['selected' + disc_name + "_profitability"] == 1]
        _data_sel = _data_sel.apply(lambda row: [row['dist_prop'], row['probs' + disc_name]], axis=1)
        _data_sel2 = []
        for el in _data_sel:
            for j in range(len(el[1])):
                _data_sel2 += [[el[0], el[1][j]]]
        _data_all = [[t[0], t[_num+1]] for t in output_list2]
        fig, ax = plt.subplots()
        plt.scatter([t[0] for t in output_list2], [t[_num+1] for t in output_list2], label=disc_lab, s=0.5)
        plt.scatter([t[0] for t in _data_sel2], [t[1] for t in _data_sel2], s=2, color='red', label='Selected')
        plt.xlabel("Mileage reduction")
        plt.ylabel("Acceptance probability")
        plt.tight_layout()
        plt.savefig('scatter_distance_acceptance_' + disc_name + str(_sample) + "." + args.pic_format, dpi=args.dpi)

    output_list = shared_reordered.apply(
        lambda row: [[row['profitability_unbalanced']] +
                     [row[t + '_profitability_unbalanced'] for t in discounts_names],
                     [row['probs']] + [row['probs_' + t] for t in discounts_names]],
        axis=1
    ).to_list()

    output_dict = {t: [] for t in discounts_labels}
    for el in output_list:
        for lab_num, label in enumerate(discounts_labels):
            for i in range(len(el[1][0])):
                output_dict[label] += [[el[0][lab_num], el[1][lab_num][i]]]

    for key in output_dict.keys():
        output_dict[key].sort(key=lambda it: it[0])

    fig, ax = plt.subplots()
    for strategy_label in discounts_labels:
        x, y = [t[0] for t in output_dict[strategy_label]], [t[1] for t in output_dict[strategy_label]]
        plt.scatter(x, y, s=0.5, label=strategy_label + " observations")
        _Poly = Polynomial.fit(x, y, 1)
        x_plot, y_plot = _Poly.linspace(n=1000, domain=[min(x), max(x)])
        plt.plot(x_plot, y_plot, lw=1.5, label=strategy_label + " regression", ls="solid")
    plt.legend(loc='lower right', fontsize=10, markerscale=3)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('scatter_profitability_acceptance_' + str(_sample) + "." + args.pic_format, dpi=args.dpi)


    _bins_x = [round(t, 1) for t in np.arange(-0.6, 0.7, 0.1)]
    _bins_y = [round(t, 1) for t in np.arange(0, 1.1, 0.1)]
    _bins_z = [round(t, 1) for t in np.arange(1, 3, 0.1)]

    def temp_foo(element, _bins):
        enum = np.digitize(element, _bins)
        if enum < len(_bins):
            return enum
        else:
            return enum - 1

    for _num, (disc_name, disc_lab) in enumerate(zip([""] + ["_" + t for t in discounts_names], discounts_labels)):
        prefix = "" if _num == 0 else disc_name[1:] + "_"
        _data_all = shared_reordered.apply(
            lambda row: [row['dist_prop'], row['probs' + disc_name], row[prefix + 'profitability_unbalanced']],
            axis=1
        )
        _data_all2 = []
        for el in _data_all:
            _data_all2 += [[el[0], np.prod(el[1]), el[2]]]
            # for j in range(len(el[1])):
            #     _data_all2 += [[el[0], el[1][j], el[2]]]

        _data_map = [[_bins_x[temp_foo(t[0], _bins_x)],
                      _bins_y[temp_foo(t[1], _bins_y)],
                      _bins_z[temp_foo(t[2], _bins_z)]] for t in _data_all2]

        _heatmap = pd.DataFrame(
            {'Relative mileage reduction': [t[0] for t in _data_map],
             'Acceptance probability': [t[1] for t in _data_map],
             'Expected profitability': [t[2] for t in _data_map]}
        ).reset_index(drop=True)
        _table = _heatmap.pivot_table(index='Relative mileage reduction',
                                      columns='Expected profitability',
                                      values='Acceptance probability',
                                      aggfunc='mean')
        for ex_prof in [round(t, 1) for t in np.arange(1, 3.1, 0.1)]:
            if ex_prof not in _table.columns:
                _table[ex_prof] = np.NaN
        _table = _table[sorted(_table.columns)]
        fig, ax = plt.subplots(figsize=(8,8))
        sns.heatmap(_table, yticklabels=[str(round(100*t, 0))[:-2] + '%' for t in _table.index],
                    cbar=True if _num == len(discounts_labels) - 1 else False,
                    cmap=sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True))
        if _num != 0:
            plt.ylabel(None)
            plt.yticks([])
            plt.xlabel(None)
        else:
            plt.ylabel('Relative mileage reduction', fontsize=25)
            plt.yticks(fontsize=25)
        if _num == len(discounts_labels) - 1:
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=25)
        plt.xticks(fontsize=15)
        plt.xlabel('Expected Profitability', fontsize=25)
        plt.tight_layout()
        plt.savefig('heatmap' + disc_name + "_" + str(_sample) + "." + args.pic_format, dpi=args.dpi)

    shared['dist_prop'] = shared.apply(
        lambda row: 1 - row['u_veh'] * args.avg_speed / sum(row['individual_distances']),
        axis=1
    )
    for baseline_name in ['jiao', 'karaenke', 'detour']:
        shared['unbalanced_profitability_' + baseline_name] = shared.apply(
            lambda x: x['profitability_' + baseline_name + '_actual']/len(x['indexes']),
            axis=1
        )
        _data_all = shared.apply(
            lambda row: [row['dist_prop'], row['prod_prob_' + baseline_name],
                         row['unbalanced_profitability_' + baseline_name]],
            axis=1
        )
        _data_map = [[_bins_x[temp_foo(t[0], _bins_x)],
                      _bins_y[temp_foo(t[1], _bins_y)],
                      _bins_z[temp_foo(t[2], _bins_z)]] for t in _data_all]

        _heatmap = pd.DataFrame(
            {'Relative mileage reduction': [t[0] for t in _data_map],
             'Acceptance probability': [t[1] for t in _data_map],
             'Expected profitability': [t[2] for t in _data_map]}
        ).reset_index(drop=True)
        _table = _heatmap.pivot_table(index='Relative mileage reduction',
                                      columns='Expected profitability',
                                      values='Acceptance probability',
                                      aggfunc='mean')
        for ex_prof in [round(t, 1) for t in np.arange(1, 3.1, 0.1)]:
            if ex_prof not in _table.columns:
                _table[ex_prof] = np.NaN
        _table = _table[sorted(_table.columns)]
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(_table, yticklabels=[str(round(100 * t, 0))[:-2] + '%' for t in _table.index],
                    cbar=True if baseline_name=='detour' else False,
                    cmap=sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True))
        if baseline_name != 'jiao':
            plt.ylabel(None)
            plt.yticks([])
            plt.xlabel(None)
        else:
            plt.ylabel('Relative mileage reduction', fontsize=25)
            plt.yticks(fontsize=25)
        if baseline_name == 'detour':
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=25)
        plt.xticks(fontsize=15)
        plt.xlabel('Expected Profitability', fontsize=25)
        plt.tight_layout()
        plt.savefig('heatmap_' + baseline_name + "." + args.pic_format, dpi=args.dpi)


if args.analysis_parts[6]:
    rr['profitability_unbalanced'] = rr.apply(
        lambda row: row['profitability'] / len(row['indexes']),
        axis=1
    )
    for disc_name in discounts_names:
        rr[disc_name + '_profitability_unbalanced'] = rr.apply(
            lambda row: row[disc_name + '_profitability'] / len(row['indexes']),
            axis=1
        )
    if args.baselines:
        for baseline_name in ['jiao', 'karaenke', 'detour']:
            rr[baseline_name + '_profitability_unbalanced'] = rr.apply(
                lambda row: row['profitability_' + baseline_name + '_actual'] / len(row['indexes']),
                axis=1
            )

    temp_data = rr.loc[rr['selected_profitability'] == 1]
    results = {
        'profitability': [np.mean(temp_data['profitability_unbalanced'])],
        'e_dist': [sum(temp_data['best_profit'].apply(lambda x: x[4]))]
    }
    for flat_disc in discounts_names:
        temp_data = rr.loc[rr['selected_' + flat_disc + '_profitability'] == 1]
        results['profitability'] += [np.mean(temp_data[flat_disc + '_profitability_unbalanced'])]
        results['e_dist'] += [sum(temp_data.apply(
            lambda x: (x['veh_dist'] * x["prod_prob_" + flat_disc] +
                       sum(x['individual_distances']) * (1 - x["prod_prob_" + flat_disc])) / 1000,
            axis=1
        ))]
    if args.baselines:
        for baseline_name in ['jiao', 'karaenke', 'detour']:
            temp_data = rr.loc[rr['selected_' + baseline_name] == 1]
            results['profitability'] += [np.mean(temp_data[baseline_name + '_profitability_unbalanced'])]
            results['e_dist'] += [sum(temp_data['baseline_' + baseline_name].apply(lambda x: x[4]))]

    results['profitability'] += [1.5]
    results['e_dist'] += [sum(singles['veh_dist']) / 1000]

    results = pd.DataFrame(results)
    if args.baselines:
        results.index = discounts_labels + ['Jiao', 'Karaenke', 'Detour'] + ['Private only']
    else:
        results.index = discounts_labels + ['Private only']
    results = results.round(2)
    print(results.to_latex(float_format="%.2f"))

