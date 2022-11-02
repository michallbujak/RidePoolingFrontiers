import warnings

import seaborn
import seaborn as sns
import utils_topology as utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import networkx as nx
import datetime
import json
from dotmap import DotMap
from netwulf import visualize
import matplotlib.ticker as mtick
import os
import tkinter as tk
import multiprocessing as mp


def create_figs_folder(config):
    try:
        os.mkdir(config.path_results + "figs")
    except OSError as error:
        print(error)
        print('overwriting current files in the folder')
    config.path_results += '/'


def config_initialisation(path, date, sblts_exmas="exmas"):
    topological_config = utils.get_parameters(path)
    topological_config.path_results = 'data/results/' + date + '/'
    topological_config.date = date
    topological_config.sblts_exmas = sblts_exmas
    return topological_config


def load_data(config):
    with open(config.path_results + '/rep_graphs_' + config.date + '.obj', 'rb') as file:
        rep_graphs = pickle.load(file)

    with open(config.path_results + '/dotmap_list_' + config.date + '.obj', 'rb') as file:
        dotmap_list = pickle.load(file)

    with open(config.path_results + '/all_graphs_list_' + config.date + '.obj', 'rb') as file:
        all_graphs_list = pickle.load(file)

    return rep_graphs, dotmap_list, all_graphs_list


def draw_bipartite_graph(graph, max_weight, config=None, save=False, saving_number=0, width_power=1,
                         figsize=(5, 12), dpi=100, node_size=1, batch_size=147, plot=True, date=None,
                         default_edge_size=1, name=None, colour_specific_node=None):
    # G1 = nx.convert_node_labels_to_integers(graph)
    G1 = graph
    x = G1.nodes._nodes
    l = []
    r = []
    for i in x:
        j = x[i]
        if j['bipartite'] == 1:
            l.append(i)
        else:
            r.append(i)

    if nx.is_weighted(G1):
        dict_weights = {tuple(edge_data[:-1]): edge_data[-1]["weight"] for edge_data in G1.edges(data=True)}
        r_weighted = {v[-1]: dict_weights[v] for v in dict_weights.keys() if v[-1] in r}
        r_weighted_sorted = {k: v for k, v in sorted(r_weighted.items(), key=lambda x: x[1])}
        r = list(r_weighted_sorted.keys())

    colour_list = len(l) * ['g'] + len(r) * ['b']

    pos = nx.bipartite_layout(G1, l)

    new_pos = dict()
    for num, key in enumerate(pos.keys()):
        if num <= batch_size - 1:
            new_pos[key] = pos[key]
        else:
            new_pos[r[num - batch_size]] = pos[key]

    if colour_specific_node is not None:
        assert isinstance(colour_specific_node, int), "Passed node number is not an integer"
        colour_list[colour_specific_node] = "r"

    plt.figure(figsize=figsize, dpi=dpi)

    nx.draw_networkx_nodes(G1, pos=new_pos, node_color=colour_list, node_size=node_size)

    if nx.is_weighted(G1):
        for weight in range(1, max_weight + 1):
            edge_list = [(u, v) for (u, v, d) in G1.edges(data=True) if d["weight"] == weight]
            nx.draw_networkx_edges(G1, new_pos, edgelist=edge_list,
                                   width=default_edge_size * np.power(weight, width_power)
                                         / np.power(max_weight, width_power))
    else:
        if colour_specific_node is None:
            nx.draw_networkx_edges(G1, new_pos, edgelist=G1.edges, width=default_edge_size)
        else:
            assert isinstance(colour_specific_node, int), "Passed node number is not an integer"
            colour_list = []
            for item in G1.edges:
                if item[0] == colour_specific_node or item[1] == colour_specific_node:
                    colour_list.append("red")
                else:
                    colour_list.append("black")
            nx.draw_networkx_edges(G1, new_pos, edgelist=G1.edges, width=default_edge_size / 5, edge_color=colour_list)

    if save:
        if date is None:
            date = str(datetime.date.today().strftime("%d-%m-%y"))
        else:
            date = str(date)
        if name is None:
            plt.savefig(config.path_results + "temp/graph_" + date + "_no_" + str(saving_number) + ".png")
        else:
            plt.savefig(config.path_results + "temp/" + name + ".png")
    if plot:
        plt.show()


def graph_visualisation_with_netwulf(all_graphs=None, rep_graphs=None, graph_list=None, show_dialogue=True):
    if graph_list is None:
        graph_list = ["single_pairs_shareability", "single_pairs_matching",
                      "full_pairs_shareability", "full_pairs_matching"]

    if all_graphs is None:
        for g in ["single_pairs_shareability", "single_pairs_matching"]:
            if g in graph_list:
                graph_list.remove(g)
    else:
        no_nodes = len(all_graphs[0]["pairs_matching"].nodes)

    if rep_graphs is None:
        for g in ["full_pairs_shareability", "full_pairs_matching"]:
            if g in graph_list:
                graph_list.remove(g)
    else:
        no_nodes = len(all_graphs[0]["pairs_matching"].nodes)

    try:
        no_nodes
    except NameError:
        warnings.warn("Error trying to read number of nodes")
    else:
        pass

    for g in graph_list:
        if show_dialogue:
            text = "The following network is " + g.upper() + \
                   ". \n Please click 'Post to python' in the browser when investigated." + \
                   "\n Default name for the graph is: " + g + "_" + str(no_nodes)
            window = tk.Tk()
            lbl = tk.Label(window, text="Input")
            lbl.pack()
            txt = tk.Text(window, width=100, height=20)
            txt.pack()
            txt.insert("1.0", text)
            button = tk.Button(window, text="Show", command=window.destroy)
            button.pack()
            window.mainloop()

        if g == "single_pairs_shareability":
            graph = all_graphs[0]['pairs_shareability']
        elif g == "single_pairs_matching":
            graph = all_graphs[0]['pairs_matching']
        elif g == "full_pairs_shareability":
            graph = rep_graphs['pairs_shareability']
        elif g == "full_pairs_matching":
            graph = rep_graphs['pairs_matching']
        else:
            raise Warning("incorrect graph_list")

        visualize(graph, config=json.load(open('data/configs/netwulf_config.json')))


def visualise_graph_evolution(dotmap_list, topological_config, num_list=None, node_size=1, dpi=80,
                              fig_size=(10, 24), plot=False, width_power=1, save=True):
    if num_list is None:
        list([1, 5, 10, 100, 900])

    for num in num_list:
        if num == 1:
            obj = [dotmap_list[1]]
        else:
            obj = dotmap_list[:num]
        draw_bipartite_graph(utils.analyse_edge_count(obj, topological_config,
                                                      list_types_of_graph=['bipartite_matching'],
                                                      logger_level='WARNING')[
                                 'bipartite_matching'],
                             num, node_size=node_size, dpi=dpi, figsize=fig_size, plot=plot, width_power=width_power,
                             config=topological_config, save=save, saving_number=num, date=topological_config.date)


def kpis_gain(dotmap_list, topological_config, max_ticks=5):
    sblts_exmas = topological_config.sblts_exmas
    str_for_end = "_" + str(len(dotmap_list[0][sblts_exmas].requests))
    multiplier = 100

    ax = sns.histplot([multiplier * (x[sblts_exmas].res.PassUtility_ns - x[sblts_exmas].res.PassUtility) / x[
        sblts_exmas].res.PassUtility_ns for x in dotmap_list])
    ax.set(xlabel="Utility gain", ylabel=None)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.locator_params(axis='x', nbins=max_ticks)
    plt.savefig(topological_config.path_results + "figs/relative_pass_utility" + str_for_end + ".png")
    plt.close()

    ax = sns.histplot([multiplier * (x[sblts_exmas].res.PassHourTrav - x[sblts_exmas].res.PassHourTrav_ns) / x[
        sblts_exmas].res.PassHourTrav_ns for x in dotmap_list])
    ax.set(xlabel="Travel time extension", ylabel=None)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.locator_params(axis='x', nbins=max_ticks)
    plt.savefig(topological_config.path_results + "figs/relative_pass_hours" + str_for_end + ".png")
    plt.close()

    ax = sns.histplot([multiplier * (x[sblts_exmas].res.VehHourTrav_ns - x[sblts_exmas].res.VehHourTrav) / x[
        sblts_exmas].res.VehHourTrav_ns for x in dotmap_list])
    ax.set(xlabel='Vehicle time shortening', ylabel=None)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.locator_params(axis='x', nbins=max_ticks)
    plt.savefig(topological_config.path_results + "figs/relative_veh_hours" + str_for_end + ".png")
    plt.close()


def probability_of_sharing_classes(dotmap_list, topological_config, name=None,
                                   _class_names=("C1", "C2", "C3", "C4"), max_bins=5):
    sblts_exmas = topological_config.sblts_exmas
    if name is None:
        name = "per_class_prob_" + str(len(dotmap_list[0][sblts_exmas].requests))

    probs = {"c0": np.array([0, 0]), "c1": np.array([0, 0]), "c2": np.array([0, 0]), "c3": np.array([0, 0])}
    for rep in dotmap_list:
        df = rep.prob.sampled_random_parameters.copy()
        df["VoT"] *= 3600
        df.set_index("new_index", inplace=True)
        c0 = df.loc[df["class"] == 0]
        c1 = df.loc[df["class"] == 1]
        c2 = df.loc[df["class"] == 2]
        c3 = df.loc[df["class"] == 3]

        schedule = rep[sblts_exmas].schedule
        indexes_sharing = schedule["indexes"].values
        a1 = {frozenset(t) for t in indexes_sharing}
        a2 = {next(iter(t)) for t in a1 if len(t) == 1}

        probs["c0"] += np.array([len(a2.intersection(set(c0.index))), len(c0)])
        probs["c1"] += np.array([len(a2.intersection(set(c1.index))), len(c1)])
        probs["c2"] += np.array([len(a2.intersection(set(c2.index))), len(c2)])
        probs["c3"] += np.array([len(a2.intersection(set(c3.index))), len(c3)])

    x = _class_names

    def foo(i):
        return 100 * (probs["c" + str(i)][1] - probs["c" + str(i)][0]) / probs["c" + str(i)][1]

    y = [foo(t) for t in range(len(x))]

    ax = sns.barplot(data=pd.DataFrame({"names": x, "values": y}), x="names", y="values")
    ax.set(ylabel='Probability of sharing', xlabel=None)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    plt.locator_params(axis='x', nbins=max_bins)
    plt.savefig(topological_config.path_results + "figs/" + name + ".png")


def amend_dotmap(dotmap, config):
    sblts_exmas = config.sblts_exmas
    df = dotmap[sblts_exmas].requests[["id", "ttrav", "ttrav_sh", "u", "u_sh", "kind"]]
    probs = dotmap.prob.sampled_random_parameters
    df = pd.merge(df, probs[["class"]], left_on="id", right_index=True)
    return df


def relative_travel_times(df):
    df = df.assign(Relative_time_add=(df["ttrav_sh"] - df["ttrav"]))
    df['Relative_time_add'] = df['Relative_time_add'].apply(lambda x: 0 if abs(x) <= 1 else x)
    df['Relative_time_add'] = df['Relative_time_add']/df['ttrav']
    return df


def separate_by_classes(list_dfs):
    classes = dict()

    first_rep = True
    for rep in list_dfs:
        for class_no in [0, 1, 2, 3]:
            if first_rep:
                classes["C" + str(class_no + 1)] = rep.loc[rep["class"] == class_no]
            else:
                classes["C" + str(class_no + 1)] =\
                    classes["C" + str(class_no + 1)].append(rep.loc[rep["class"] == class_no])
        first_rep = False

    return classes


def individual_analysis(dotmap_list, config, percentile=95):
    results = [amend_dotmap(indata, config) for indata in dotmap_list]
    results = [relative_travel_times(df) for df in results]

    maximal_delay_percentile = np.percentile(pd.concat(results, axis=0)["Relative_time_add"], percentile)

    results_shared = [df.loc[df["kind"] > 1] for df in results]

    classes_shared = separate_by_classes(results_shared)

    datasets = [t["Relative_time_add"] for t in [v for k, v in classes_shared.items()]]
    labels = [k for k, v in classes_shared.items()]

    for data, _label in zip(datasets, labels):
        plt.hist(data, density=True, stacked=True, histtype='step', alpha=0.6, bins=50, label=_label)
    plt.legend(loc="upper right")
    plt.show()

    z = 0
