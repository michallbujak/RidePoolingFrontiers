import osmnx as ox
import networkx as nx
import matplotlib.pylab as plt

graph = ox.load_graphml('graph_skotniki.graphml')
# nx.draw_networkx(graph, node_size=10, with_labels=False)
ox.plot_graph(graph)
plt.show()
x = 0
