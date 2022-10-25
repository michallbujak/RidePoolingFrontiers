import pickle
import networkx as nx
import pandas as pd
from netwulf import visualize
import json
import numpy as np
from utils_topology import draw_bipartite_graph
import utils_topology as utils
import matplotlib.pyplot as plt
import datetime
import matplotlib.image as mpimg
import netwulf
import seaborn as sns
import matplotlib.ticker as mtick

date = "22-10-22"
special_name = ""
sblts_exmas = "exmas"

# with open('data/results/' + date + '/rep_graphs_' + date + '.obj', 'rb') as file:
#     e = pickle.load(file)

with open('data/results/' + date + '/dotmap_list_' + date + '.obj', 'rb') as file:
    e = pickle.load(file)

# with open('data/results/' + date + '/all_graphs_list_' + date + '.obj', 'rb') as file:
#     e = pickle.load(file)

topological_config = utils.get_parameters('data/configs/topology_settings3.json')
# utils.create_results_directory(topological_config, date=date)
topological_config.path_results = 'data/results/' + date + special_name + '/'

# G = e[0]['pairs_matching']
# G = e['pairs_shareability']

# colours = []
# for item in G.nodes:
#     if item == 0:
#         colours.append("red")
#     else:
#         colours.append("black")
# nx.set_node_attributes(G, dict(zip(list(G.nodes), colours)), "group")
# visualize(G, config=json.load(open('data/configs/netwulf_config.json')))
#
# edge_colours = []
# for item in G.edges:
#     if item[0] == 0 or item[1] == 0:
#         edge_colours.append("red")
#     else:
#         edge_colours.append("black")
# nx.set_edge_attributes(G, dict(zip(list(G.edges), colours)), "colour")
#
# visualize(G, config=json.load(open('data/configs/netwulf_config.json')))

# nx.draw_networkx_nodes(G, pos=nx.spring_layout(G), node_size=10, node_color="black")
# nx.draw_networkx_edges(G, pos=nx.spring_layout(G), width=2, edge_color="black")
# plt.show()


# visualize(e['pairs_matching'], config=json.load(open('data/configs/netwulf_config.json')))
# draw_bipartite_graph(e['bipartite_matching'], 1000, topological_config, date=date, save=True,
#                      name="full_bi_matching", dpi=200, colour_specific_node=None,
#                      default_edge_size=0.1)

# num_list = [1, 5, 10, 100, 900]
# for num in num_list:
#     if num == 1:
#         obj = [e[1]]
#     else:
#         obj = e[:num]
#     draw_bipartite_graph(utils.analyse_edge_count(obj, topological_config,
#                                                   list_types_of_graph=['bipartite_matching'], logger_level='WARNING')[
#                              'bipartite_matching'],
#                          num, node_size=1, dpi=80, figsize=(10, 24), plot=False, width_power=1,
#                          config=topological_config, save=True, saving_number=num, date=date)

# fig, axes = plt.subplots(nrows=2, ncols=5, sharex='col', sharey='row')
#
# for col in range(2):
#     for row in range(5):
#         num = num_list[5*col+row]
#         img = mpimg.imread(topological_config.path_results + "temp/graph_" +
#                            str(datetime.date.today().strftime("%d-%m-%y")) + "_no_" + str(num) + ".png")
#         axes[col, row].imshow(img)
#         axes[col, row].set_title('Step ' + str(num))
#         axes[col, row].axis('off')
#
# plt.savefig(topological_config.path_results + "graph_growth" + str(datetime.date.today().strftime("%d-%m-%y")) + ".png")
# plt.show()

# topological_config.path_results = 'data/results/31-05-22/'

# num_list = [1] + list(range(100, 1000, 100))
# df = pd.DataFrame()
# for num in num_list:
#     if num == 1:
#         obj = [e[0]]
#     else:
#         obj = e[:num]
#     temp_df = utils.analysis_all_graphs(obj, topological_config, save=False, save_num=num, date='31-05-22')
#     df = pd.concat([df, temp_df])
#
# df.to_excel(topological_config.path_results + 'all_graphs_properties_' + '31-05-22' + '.xlsx')

# num_list = list(range(1000))
# df = pd.DataFrame()
# for num in num_list:
#     if num == 0:
#         obj = [e[0]]
#     else:
#         obj = e[:num]
#     temp_graph = utils.analyse_edge_count(obj, topological_config, list_types_of_graph=['pairs_matching'],
#                              logger_level='WARNING')['pairs_matching']
#     t = utils.graph_mini_graphstatistics(temp_graph)
#     temp_df = pd.DataFrame.from_dict({'average_degree': [t.average_degree], 'max_comp': [t.proportion_max_component],
#                                       'number_of_isolated': [t.number_of_isolated_pairs]})
#     df = pd.concat([df, temp_df])
#
# df.reset_index(inplace=True)
# df.drop(columns=['index'], inplace=True)
# df.to_excel(topological_config.path_results + 'frame_evolution_06-06-22.xlsx', index=False)

# str_for_end = "_198"
# ticks = 5
# multiplier = 100
#
# ax = sns.histplot([multiplier*(x[sblts_exmas].res.PassUtility_ns - x[sblts_exmas].res.PassUtility) / x[sblts_exmas].res.PassUtility_ns for x in e])
# ax.set(xlabel="Utility gain", ylabel=None)
# ax.xaxis.set_major_formatter(mtick.PercentFormatter())
# plt.locator_params(axis='x', nbins=ticks)
# plt.savefig(topological_config.path_results + "figs/relative_pass_utility" + str_for_end + ".png")
# plt.close()
#
# ax = sns.histplot([multiplier*(x[sblts_exmas].res.PassHourTrav - x[sblts_exmas].res.PassHourTrav_ns) / x[sblts_exmas].res.PassHourTrav_ns for x in e])
# ax.set(xlabel="Travel time extension", ylabel=None)
# ax.xaxis.set_major_formatter(mtick.PercentFormatter())
# plt.locator_params(axis='x', nbins=ticks)
# plt.savefig(topological_config.path_results + "figs/relative_pass_hours" + str_for_end + ".png")
# plt.close()
#
# ax = sns.histplot([multiplier*(x[sblts_exmas].res.VehHourTrav_ns - x[sblts_exmas].res.VehHourTrav) / x[sblts_exmas].res.VehHourTrav_ns for x in e])
# ax.set(xlabel='Vehicle time shortening', ylabel=None)
# ax.xaxis.set_major_formatter(mtick.PercentFormatter())
# plt.locator_params(axis='x', nbins=ticks)
# plt.savefig(topological_config.path_results + "figs/relative_veh_hours" + str_for_end + ".png")
# plt.close()


# y = [(x[sblts_exmas].res.PassUtility_ns - x[sblts_exmas].res.PassUtility) / x[sblts_exmas].res.PassUtility_ns for x in e]
# print(y)
# x = 0

# with open("C:/Users/szmat/Documents/GitHub/ExMAS_sideline/Topology/data/results/"
#           + date + "/final_matching_" + date + ".json", "r") as file:
#     matches = json.load(file)
#
# number_elements = 99
# replications = 100
#
# # match_list = [list(map(lambda x: x.replace("(", "").replace(")", ""),
# #                        list(matches.keys())[t].split(","))) for t in range(len(matches.keys()))]
# #
# # list(map(lambda x: x.pop(-1) if x[-1] == "" else x, match_list))
#
# str_for_end = "_" + str(number_elements)
# ticks = 5
#
# sharing_prob = []
#
# for j in range(number_elements):
#     sharing_prob.append((replications - matches.get("(" + str(j) + ",)", 0))*100/replications)
#
#
# ax = sns.histplot(sharing_prob)
# ax.set(xlabel='Probability of sharing', ylabel=None)
# ax.xaxis.set_major_formatter(mtick.PercentFormatter())
# plt.locator_params(axis='x', nbins=ticks)
# plt.xlim(0, 100)
# plt.savefig(topological_config.path_results + "figs/prob_sharing" + str_for_end + ".png")
# plt.close()


probs = {"c0": np.array([0, 0]), "c1": np.array([0, 0]), "c2": np.array([0, 0]), "c3": np.array([0, 0])}
for rep in e:
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

x = ("c0", "c1", "c2", "c3")


def foo(i):
    return (probs["c" + str(i)][1] - probs["c" + str(i)][0]) / probs["c" + str(i)][1]


y = [foo(t) for t in range(4)]

sns.barplot(data=pd.DataFrame({"names": x, "values": y}), x="names", y="values")
plt.show()
