import pickle
import networkx as nx
import netwulf as nw
import matplotlib.pyplot as plt


G = nx.barabasi_albert_graph(120, 4)

for edge in G.edges:
       nx.set_edge_attributes(G, {edge: {"color": "red"}})

stylized_network, config = nw.visualize(G)

# fig, ax = nw.tools.draw_netwulf(stylized_network)
node_pos_list = {i: nw.tools.node_pos(stylized_network, i) for i in range(len(list(G.nodes)))}


# plt.show()

# nx.draw_networkx_nodes(G, pos=node_pos_list)
# layout = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos=node_pos_list, node_color="black")
nx.draw_networkx_edges(G, node_pos_list)
plt.tight_layout()
plt.show()

x = 0
