"""
Implementation of GNN model.
"""
import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric import utils as geo_utils
from torch_geometric.nn import DMoNPooling, GCNConv, Sequential, dense_mincut_pool

import numpy as np
import utils_ml as utils
from dmon import DMoNPooling


class GNN(torch.nn.Module):
    """
    Based on https://github.com/FilippoMB/Simplifying-Clustering-with-Graph-Neural-Networks.
    """

    def __init__(
            self,
            mp_units,
            mp_act,
            in_channels,
            n_clusters,
            mlp_units,
            mlp_act,
            dmon,
            adj,
            kappa,
            gumbel_tau,
    ):
        super().__init__()

        self.kappa = kappa
        self.gumbel_tau = gumbel_tau
        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)

        if len(mp_units) > 0:
            header = "x, edge_index, edge_weight -> x"
            gcn_layer = GCNConv(in_channels, mp_units[0], normalize=False)
            mp = [(gcn_layer, header), mp_act]
            for i in range(len(mp_units) - 1):
                gcn_layer = GCNConv(mp_units[i], mp_units[i + 1], normalize=False)
                mp.append((gcn_layer, header))
                if (i < len(mp_units) - 2) or (len(mlp_units) > 0) or dmon:
                    mp.append(mp_act)
            self.mp = Sequential("x, edge_index, edge_weight", mp)
            out_chan = mp_units[-1]
        else:
            self.mp = torch.nn.Sequential()
            out_chan = in_channels

        self.mlp = torch.nn.Sequential()
        for i, units in enumerate(mlp_units):
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            if i < len(mlp_units) - 1:
                self.mlp.append(mlp_act)
            out_chan = units

        if dmon:
            self.dmon = DMoNPooling(out_chan, n_clusters, gumbel_tau=gumbel_tau)
            out_chan = n_clusters
        else:
            self.dmon = None
        assert out_chan == n_clusters

        if adj is not None:
            self.r = torch.nn.Parameter(adj.clone().detach().requires_grad_(True))
        else:
            self.r = None

    def forward(self, x, edge_index, edge_weight, components):
        # Propagate node feats
        x = self.mp(x, edge_index, edge_weight)

        # Cluster assignments (logits)
        s = self.mlp(x)

        # Compute loss
        adj = geo_utils.to_dense_adj(
            edge_index, edge_attr=edge_weight, max_num_nodes=x.shape[0]
        )[0]
        initial_adj = adj.clone()
        adj[adj <= 0] = 0.

        # DMoN losses
        if self.dmon is None:
            if self.gumbel_tau is not None:
                soft = F.gumbel_softmax(s, dim=1, tau=self.gumbel_tau)
            else:
                soft = torch.softmax(s, dim=-1)
            spectral_loss, ortho_loss, cluster_loss = (
                torch.zeros([]),
                torch.zeros([]),
                torch.zeros([]),
            )
        else:
            # adj[adj < 0] = 0.
            soft, _, _, spectral_loss, ortho_loss, cluster_loss = self.dmon(
                x.unsqueeze(0), adj.unsqueeze(0)
            )

            soft = soft[0]

        # Travel time loss using r
        if (self.r is not None) and (self.kappa[3] != 0):
            r = torch.softmax(self.r, dim=-1)
            q = r * r.T
            p = torch.stack([(soft * soft[idx]).sum(-1) for idx in range(q.shape[0])])
            w = p * q
            I = torch.eye(adj.shape[0])
            adj_wth_loops = adj * (1 - I.float())
            adj_wth_loops_bool = (adj_wth_loops > 1e-8).float()
            trav_time_loss = -(
                    (w * adj_wth_loops).sum(dim=-1)
                    + (1 - (w * adj_wth_loops_bool).sum(dim=-1)) * adj[I.bool()]
            ).sum()
        else:
            trav_time_loss = torch.zeros([])

        # Balance pooling loss from Simplifying clustering...
        if self.kappa[4] != 0:
            _, _, balance_pool_loss = utils.just_balance_pool(x, adj, s)
        else:
            balance_pool_loss = torch.zeros([])

        # Our classification loss
        if self.kappa[5] != 0:
            our_classification_loss = -(soft ** 2).sum()
        else:
            our_classification_loss = torch.zeros([])

        # Our entropy loss
        if self.kappa[6] != 0:
            our_entropy_loss = (soft * torch.log(soft)).sum()
        else:
            our_entropy_loss = torch.zeros([])

        # Our weighted MSE loss
        if self.kappa[7] != 0:
            our_mse_loss = torch.stack(
                [
                    torch.matmul(initial_adj[idx], torch.sum((soft - soft[idx]) ** 2, dim=1))
                    for idx in range(adj.shape[-1])
                ]
            ).mean()
        else:
            our_mse_loss = torch.zeros([])

        if self.kappa[8] != 0:

            adj_wth_loops_bool = (
                    (torch.abs(adj) * (1 - torch.eye(adj.shape[0], dtype=float))) > 1e-8
            ).float()
            b = adj_wth_loops_bool.unsqueeze(0) * torch.bmm(
                soft.T.unsqueeze(-1), soft.T.unsqueeze(1)
            )
            degree_loss = -torch.log(b.sum(dim=(1, 2))).sum()
        else:
            degree_loss = torch.zeros([])

        if self.kappa[9] != 0:

            adj_wth_loops = ((adj.float() * (1 - torch.eye(adj.shape[0], dtype=float))) > 0).float()
            sas = torch.mm(torch.mm(soft.T, adj_wth_loops).T, soft.T) + torch.eye(adj.shape[0])
            unique, counts = np.unique(components, return_counts=True)

            epidemic_thr_loss = []
            N = adj.shape[0]
            for c, count in zip(unique, counts):
                if count == 1: continue

                sas_sub = sas[components == c][:, components == c]
                up = torch.trace(sas_sub)
                down = (torch.diagonal(sas_sub) ** 2).sum()
                thr = (count / N) * (up / down)
                epidemic_thr_loss += [thr]
            epidemic_thr_loss = -sum(epidemic_thr_loss)
        else:
            epidemic_thr_loss = torch.zeros([])

        if self.kappa[10] != 0 or self.kappa[11] != 0:
            _, _, mincut_mc_loss, mincut_orth_loss = dense_mincut_pool(x, adj, s)
        else:
            mincut_mc_loss, mincut_orth_loss = torch.zeros([]), torch.zeros([])

        loss = {
            "spectral": spectral_loss,
            "ortho": ortho_loss,
            "cluster": cluster_loss,
            "trav_time": trav_time_loss,
            "balance_pool": balance_pool_loss,
            "our_classification": our_classification_loss,
            "our_entropy": our_entropy_loss,
            "our_mse": our_mse_loss,
            "degree": degree_loss,
            "epidemic_thr": epidemic_thr_loss,
            "mincut_mc": mincut_mc_loss,
            "mincut_orth": mincut_orth_loss,
        }
        return soft, loss