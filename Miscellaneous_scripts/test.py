from ExMAS.main_prob import main as exmas_algo
from NYC_tools import NYC_data_prep_functions as nyc_func

import matplotlib.pyplot as plt
import networkx as nx

G1 = nx.Graph()
travellers = ['Pass_A', 'Pass_B', 'Pass_C', 'Pass_D']
rides = ['Ride_A', 'Ride_B', 'Ride_C', 'Ride_D', 'Ride_AB', 'Ride_AD']
G1.add_nodes_from(travellers, bipartite=1)
G1.add_nodes_from(rides, bipartite=0)
edges = [('Pass_A', 'Ride_A'), ('Pass_B', 'Ride_B'), ('Pass_C', 'Ride_C'), ('Pass_D', 'Ride_D'),
         ('Pass_A', 'Ride_AB'), ('Pass_B', 'Ride_AB'), ('Pass_A', 'Ride_AD'), ('Pass_D', 'Ride_AD')]

x = G1.nodes._nodes
l = []
r = []
for i in x:
    j = x[i]
    if j['bipartite'] == 1:
        l.append(i)
    else:
        r.append(i)

r.sort(key=lambda x: len(x))
colour_list = len(l) * ['g'] + len(r) * ['b']

pos = nx.bipartite_layout(G1, l)

plt.figure()

nx.draw_networkx_nodes(G1, pos=pos, node_color=colour_list, node_size=100, label=True)
nx.draw_networkx_edges(G1, pos, edgelist=edges, width=1)

plt.show()

