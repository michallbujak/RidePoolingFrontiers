import numpy as np

from ExMAS.main_prob import main as exmas_algo
from NYC_tools import NYC_data_prep_functions as nyc_func

import matplotlib.pyplot as plt
import networkx as nx

G1 = nx.Graph()
travellers = ['__A', '__B', '__C', '__D']
rides = ['_A', '_B', '_C', '_D', '_AB', '_AD']
G1.add_nodes_from(travellers, bipartite=1)
G1.add_nodes_from(rides, bipartite=0)
edges = [('__A', '_A'), ('__B', '_B'), ('__C', '_C'), ('__D', '_D'),
         ('__A', '_AB'), ('__B', '_AB'), ('__A', '_AD'), ('__D', '_AD')]

x = G1.nodes._nodes
l = []
r = []
for i in x:
    j = x[i]
    if j['bipartite'] == 1:
        l.append(i)
    else:
        r.append(i)

colour_list = len(l) * ['g'] + len(r) * ['b']

r.sort(key=lambda x: len(x), reverse=True)

pos = nx.bipartite_layout(G1, l)

new_pos = dict()

new_pos['__A'] = np.array([-1., 0.625])
new_pos['__B'] = np.array([-1., 0.20833333])
new_pos['__C'] = np.array([-1., -0.20833333])
new_pos['__D'] = np.array([-1., -0.625])
new_pos['_A'] = np.array([0.66666667, 0.625])
new_pos['_B'] = np.array([0.66666667, 0.375])
new_pos['_C'] = np.array([0.66666667, 0.125])
new_pos['_D'] = np.array([0.66666667, -0.125])
new_pos['_AB'] = np.array([0.66666667, -0.375])
new_pos['_AD'] = np.array([0.66666667, -0.625])

labels1 = dict()
for node in G1.nodes():
    if node in travellers:
        labels1[node] = node
labels2 = dict()
for node in G1.nodes():
    if node in rides:
        labels2[node] = node

plt.figure()

nx.draw_networkx_nodes(G1, pos=new_pos, node_size=30, nodelist=travellers, node_color='black')
nx.draw_networkx_nodes(G1, pos=new_pos, node_size=30, nodelist=rides, node_color='black', node_shape='s')
# nx.draw_networkx_labels(G1, new_pos, labels1, font_size=10, font_color='r')
# nx.draw_networkx_labels(G1, new_pos, labels2, font_size=10, font_color='r')
edges1 = [('__A', '_A'), ('__B', '_B'), ('__C', '_C'), ('__D', '_D')]
nx.draw_networkx_edges(G1, new_pos, edgelist=edges1, edge_color='g')
edges2 = [('__A', '_AB'), ('__B', '_AB'), ('__A', '_AD'), ('__D', '_AD')]
nx.draw_networkx_edges(G1, new_pos, edgelist=edges2, edge_color='b')

plt.show()
