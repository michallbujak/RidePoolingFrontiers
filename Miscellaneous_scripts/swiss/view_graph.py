import networkx as nx
import os
import matplotlib.pyplot as plt

graph = nx.read_graphml('../output/switzerland_major_cities_multigraph.graphml')

print(graph.edges(data=True))
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

png_path = os.path.join(OUTPUT_DIR, "swiss_graph.png")

# your plotting code
plt.figure(figsize=(12, 12))
pos = {node: (data["lon"], data["lat"]) for node, data in graph.nodes(data=True)}
nx.draw(graph, pos,
        with_labels=True,
        node_size=100,
        font_size=6,
        edge_color="gray",
        node_color="steelblue")

plt.savefig(png_path, dpi=300)
plt.close()

print(f"Graph image saved to: {png_path}")

