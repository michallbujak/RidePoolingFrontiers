import pandas as pd
import networkx as nx
import numpy as np
from typing import List
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.colors as mcols

from matplotlib.backends.backend_agg import FigureCanvasAgg

import scipy.ndimage as ndimage
import io


def str_to_list(s: str, dtype=int) -> List:
    s = s.replace("[", "").replace("]", "")
    if not s:
        return []
    return [dtype(t) for t in s.split(",")]


def get_cluster(clusters: pd.DataFrame, idx: int) -> int:
    return clusters.iloc[idx].cluster


rides = pd.read_csv("rides.csv", index_col=0)
requests = pd.read_csv("requests.csv", index_col=0)
clusters = pd.read_csv("out.csv", index_col=0)
rides["indexes"] = rides.apply(lambda x: str_to_list(x["indexes"]), axis=1)

city_G = ox.load_graphml(
    filepath=r"C:\Users\szmat\Documents\GitHub\transportation-graph-optimization\data\exmas_nyc_data\Manhattan.graphml")

colours = list(mcols.BASE_COLORS.keys())

colour_map = {k: colours[int(v)] for k, v in clusters['cluster'].to_dict().items()}

city_nodes_dict = dict(city_G.nodes(data=True))

requests = requests.reset_index()

fig, ax = ox.plot_graph(city_G, figsize=(8, 8), node_size=0, edge_linewidth=0.3,
                        show=False, close=False,
                        edge_color='grey', bgcolor='white')


def scatter_node(node, color_map, m_size=4):
    ax.scatter(node['x_org'], node['y_org'], s=m_size, c=color_map[node['index']], marker='o')


requests.apply(scatter_node, color_map=colour_map, axis=1)
plt.show()



