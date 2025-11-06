"""
Script for training.
"""
import argparse

from collections import Counter

import pandas as pd
import torch
import networkx as nx
import numpy as np

from torch_geometric import utils as geo_utils

import utils_ml as utils

from model_gnn import GNN

parser = argparse.ArgumentParser()
parser.add_argument("--requests-csv", type=str, required=True)
parser.add_argument("--rides-csv", type=str, required=True)
parser.add_argument("--output-csv", type=str, required=True)
parser.add_argument("--delta", type=float, default=0.85)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--num-clusters", type=int, default=3)
parser.add_argument("--mp-units", type=str, default="[16, 16]")
parser.add_argument("--mlp-units", type=str, default="[3]")
parser.add_argument("--dmon", action="store_true")
parser.add_argument("--include-edge-weight", action="store_true")
parser.add_argument("--include-node-features", action="store_true")
parser.add_argument("--kappa", type=str, default="[0.,0.,0.,0.,0.,1.,1.,0,0,0]")
parser.add_argument("--pred-epoch", type=int, default=100)
parser.add_argument("--add-preds", action="store_true")
parser.add_argument("--weight-power", type=float, default=1)
parser.add_argument("--laplacian", action="store_true")
parser.add_argument("--edge-weight", type=str, choices=["u", "frac_u"])
parser.add_argument("--gumbel-tau", type=float, default=None)
args = parser.parse_args()
print(args)

mp_units = utils.str_to_list(args.mp_units)
mlp_units = utils.str_to_list(args.mlp_units)
kappa = utils.str_to_list(args.kappa, dtype=float)

assert len(kappa) == 12 + int(args.add_preds)


requests = pd.read_csv(args.requests_csv, index_col=0)
rides = pd.read_csv(args.rides_csv, index_col=0)
rides["indexes"] = rides.apply(lambda x: utils.str_to_list(x["indexes"]), axis=1)

G = utils.build_shareability_graph(requests, rides)

G_positive = G.copy()
negative_edges = list(filter(lambda e: e[2] <= 0, (e for e in G.edges.data(args.edge_weight))))
le_ids = list(e[:2] for e in negative_edges)
G_positive.remove_edges_from(le_ids)
#exit(0)
assert len(G.nodes) == len(G_positive.nodes)
comps = nx.connected_components(G_positive.to_undirected())
components = np.zeros(len(G.nodes), dtype=int)
for i, c in enumerate(comps):
    for e in c:
        components[e] = i
print(f"Nodes: {len(G.nodes)} | Edges: {len(G.edges)}")

torch.manual_seed(1)
torch.cuda.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.include_node_features:
    x = utils.get_features(requests).to(device)
else:
    x = torch.eye(len(G.nodes)).to(device)

delta = args.delta
num_clusters = args.num_clusters

adj_matrix = nx.adjacency_matrix(G, weight=args.edge_weight)

edge_weight = torch.empty(len(G.edges)) if args.include_edge_weight else None
edge_index = torch.empty((2, len(G.edges)), dtype=torch.int64)
shift_factor = min(np.min(adj_matrix), 0) if args.laplacian else 0
for i, e in enumerate(G.edges):
    edge_index[0, i] = e[0]
    edge_index[1, i] = e[1]
    if args.include_edge_weight:
        #if args.laplacian:
        edge_weight[i] = max(adj_matrix[e], 0)
        edge_weight[i] = adj_matrix[e]
        #else:
        #    edge_weight[i] = (adj_matrix[e] - shift_factor) ** args.weight_power

original_adj = geo_utils.to_dense_adj(edge_index, edge_attr=edge_weight)
if args.laplacian:
    edge_index, edge_weight = geo_utils.get_laplacian(
        edge_index, edge_weight, normalization="sym"
    )
    L = geo_utils.to_dense_adj(edge_index, edge_attr=edge_weight)[0]
    A = torch.eye(L.shape[-1]) - delta * L
else:
    A = original_adj + (1 - delta) * torch.eye(original_adj.shape[-1])

edge_index, edge_weight = geo_utils.dense_to_sparse(A)
edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)
model = GNN(
    mp_units,
    "ReLU",
    x.shape[1],
    num_clusters,
    mlp_units,
    "ReLU",
    args.dmon,
    A if kappa[3] != 0 else None,
    kappa,
    args.gumbel_tau,
)

model = model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train(model, optimizer, x, edge_index, edge_weight, components, preds, mask, kappa):
    model.train()
    optimizer.zero_grad()
    clusters, loss = model(x, edge_index, edge_weight, components)
    tot_loss = torch.matmul(
        torch.stack(list(loss.values())), torch.tensor(kappa[: len(loss)])
    )
    if torch.any(mask):
        p = preds[mask]
        c = clusters[mask]
        pred_loss = torch.nn.CrossEntropyLoss()(c, p)
        tot_loss += kappa[-1] * pred_loss
        loss["pred_loss"] = pred_loss
    tot_loss.backward()
    loss["total"] = tot_loss
    optimizer.step()
    return loss


@torch.no_grad()
def test(model, x, edge_index, edge_weight, components):
    model.eval()
    softmax, _ = model(x, edge_index, edge_weight, components)
    return softmax.argmax(axis=1).cpu(), softmax


def add_preds(model, x, edge_index, edge_weight, components, preds, mask):
    clusters, _ = model(x, edge_index, edge_weight, components)
    rang = torch.arange(0, x.shape[0], 1).long()
    to_set = rang[~mask]
    clusters_max = clusters.amax(axis=1)
    indexes = torch.argsort(clusters_max, descending=True)
    indexes = list(indexes)
    [indexes.remove(_) for _ in list(rang[mask])]
    indexes = torch.tensor(indexes[: int(clusters.shape[0] * 0.1)]).long()
    mask[indexes] = True
    preds[indexes] = clusters.argmax(axis=1)[indexes]
    return preds, mask


preds = -torch.ones(x.shape[0]).to(device).long()
mask = torch.zeros_like(preds).to(device).bool()

for epoch in range(1, args.epochs + 1):
    train_loss = train(model, optimizer, x, edge_index, edge_weight, components, preds, mask, kappa)
    clusters, _ = test(model, x, edge_index, edge_weight, components)
    clusters = dict(Counter(clusters.numpy()))

    if args.add_preds and (epoch % args.pred_epoch == 0) and (not torch.all(mask)):
        preds, mask = add_preds(model, x, edge_index, edge_weight, components, preds, mask)
        print(f"Predicted {mask.sum()}/{mask.shape[0]}")

    if epoch % 10 == 0:
        train_loss = {
            k: round(v.item(), 3) for i, (k, v) in enumerate(train_loss.items())
        }
        print(f"Epoch: {epoch:03d}, Loss: {train_loss['total']} | {clusters}")
        print(
            {
                k: v
                for i, (k, v) in enumerate(train_loss.items())
                if k != "total" and kappa[i] != 0
            }
        )

clusters, softmax = test(model, x, edge_index, edge_weight, components)
out = {"cluster": clusters}
for c in range(args.num_clusters):
    out[f"soft_{c}"] = softmax[:, c]
pd.DataFrame(out).to_csv(args.output_csv)