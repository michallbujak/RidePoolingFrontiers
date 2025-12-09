import argparse

from collections import Counter

import pandas as pd
import torch
import networkx as nx
import numpy as np
import osmnx as ox
from numpy.random import gumbel

from torch_geometric import utils as geo_utils

import utils_ml as utils

from dmon import DMoNPooling

parser = argparse.ArgumentParser()
parser.add_argument("--graph-path", type=str, required=True)
parser.add_argument("--output-csv", type=str, required=True)
parser.add_argument("--delta", type=float, default=0.85)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--num-clusters", type=int, default=8)
args = parser.parse_args()
print(args)

# read data
graph = ox.load_graphml(args.graph_path)

# prep device
torch.manual_seed(1)
torch.cuda.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
delta = args.delta
num_clusters = args.num_clusters

# get node mapping: map original node IDs → 0, 1, ..., N-1
nodes = list(graph.nodes())
node_to_idx = {node: i for i, node in enumerate(nodes)}

# build edge_index
edge_list = []
for u, v in graph.edges():
    edge_list.append([node_to_idx[u], node_to_idx[v]])
    if not graph.is_directed():
        edge_list.append([node_to_idx[v], node_to_idx[u]])  # undirected

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # shape: [2, E]
edge_index = edge_index.to(device)

propagation_vector = torch.eye(len(graph.nodes)).to(device)
# adj_matrix = nx.adjacency_matrix(graph)
original_adj = geo_utils.to_dense_adj(edge_index)
modified_adjacency = original_adj + (1 - delta) * torch.eye(original_adj.shape[-1]).to(device)
# adj_matrix = geo_utils.dense_to_sparse(modified_adjacency)
adj_matrix = modified_adjacency.to(device)

model = DMoNPooling(
    channels=propagation_vector.shape[1],
    k=args.num_clusters,
    dropout=0.0,
    gumbel_tau=None
)

model = model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
x = torch.eye(len(graph.nodes)).to(device)

def train(model, optimizer, x, adj_matrix):
    model.train()
    optimizer.zero_grad()
    s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = model(x, adj_matrix)
    tot_loss = torch.add(spectral_loss, ortho_loss)
    tot_loss = torch.add(tot_loss, cluster_loss)
    # tot_loss = np.sum([spectral_loss, ortho_loss, cluster_loss])
    tot_loss.backward()
    optimizer.step()
    return tot_loss, {'spectral_loss': spectral_loss,
                      'ortho_loss': ortho_loss,
                      'cluster_loss': cluster_loss}

@torch.no_grad()
def test(model, x, adjacency_matrix):
    model.eval()
    softmax, o, o_a, l1, l2, l3 = model(x, adjacency_matrix, None)
    return softmax.argmax(axis=2).cpu(), softmax

for epoch in range(1, args.epochs + 1):
    train_loss, loss_comps = train(model, optimizer, x, adj_matrix)
    clusters, _ = test(model, x, adj_matrix)
    clusters = dict(Counter(clusters.numpy()[0]))

    if epoch % 20 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {train_loss} | {clusters}")
        # print(
        #     {
        #         k: v
        #         for i, (k, v) in enumerate(loss_comps.items())
        #     }
        # )

clusters, softmax = test(model, x, adj_matrix)
out = {"cluster": clusters}
for c in range(args.num_clusters):
    out[f"soft_{c}"] = softmax[:, c]
pd.DataFrame(out['cluster'][0]).to_csv(args.output_csv)