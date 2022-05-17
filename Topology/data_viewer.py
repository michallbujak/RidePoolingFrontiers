from utils_topology import draw_bipartite_graph
import pickle
import networkx as nx
from netwulf import visualize
import json
import numpy as np
from matplotlib import pyplot as plt

with open('data/results/17-05-22/rep_graphs.obj', 'rb') as file:
    e = pickle.load(file)

graph = e['bipartite_matching']



draw_bipartite_graph(graph, 4)
# G1 = nx.convert_node_labels_to_integers(graph)
# x = G1.nodes._nodes
# l = []
# r = []
# for i in range(len(x)):
#     j = x[i]
#     if j['bipartite'] == 1:
#         l.append(i)
#     else:
#         r.append(i)
# colour_list = len(l) * ['r'] + len(r) * ['b']
# pos = nx.bipartite_layout(G1, l)
# # nx.draw(G1, pos=pos, with_labels=False, node_color=colour_list, node_size=2)
#
# # plt.style.use('seaborn-whitegrid')
# plt.figure(figsize=(3, 12), dpi=100)
# nx.draw_networkx_nodes(G1, pos=pos, node_color=colour_list, node_size=1)
# rep_no = 4
# for k in range(rep_no + 1):
#     edge_list = [(u, v) for (u, v, d) in G1.edges(data=True) if d["weight"] == k]
#     nx.draw_networkx_edges(G1, pos, edgelist=edge_list, width=k/20)
#
#
# plt.show()
# visualize(G1, config=json.load(open('data/configs/netwulf_config.json')))
