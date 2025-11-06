"""
Utility methods.
"""
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
import platform
import pulp
import torch


def str_to_list(s: str, dtype=int) -> List:
    s = s.replace("[", "").replace("]", "")
    if not s:
        return []
    return [dtype(t) for t in s.split(",")]


def build_shareability_graph(
        requests: pd.DataFrame,
        rides: pd.DataFrame,
        directed: bool = True
) -> nx.Graph:
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(requests.index)
    edges = []
    _rides = rides.copy()
    times = {}
    for _, row in rides.iterrows():
        idx = row["indexes"]
        if len(idx) < 2:
            times[str(idx)] = row["u_veh"]
        if len(idx) == 2:
            rev = idx[::-1]
            if (str(idx) not in times) and (str(rev) not in times):
                times[str(idx)] = row["u_veh"]
            if str(rev) in times:
                times[str(rev)] = max(times[str(rev)], row["u_veh"])

    for _, row in _rides.iterrows():
        if len(row.indexes) == 2:
            e = row.indexes
            if str(e) not in times:
                continue
            a, b, c = [e[0]], [e[1]], list(e)
            a = times[str(a)]
            b = times[str(b)]
            c = times[str(c)]
            if directed:
                edges.append(
                    (
                        row.indexes[0],
                        row.indexes[1],
                        {
                            "u": row.PassSecTrav_ns - row.u_veh,
                            "frac_u": (a + b - c) / (a + b),
                        },
                    )
                )
                edges.append(
                    (
                        row.indexes[1],
                        row.indexes[0],
                        {
                            "u": row.PassSecTrav_ns - row.u_veh,
                            "frac_u": (a + b - c) / (a + b),
                        },
                    )
                )
            else:
                edges.append(
                    (
                        row.indexes[0],
                        row.indexes[1],
                        {
                            "u": a + b - c,
                            "frac_u": (a + b - c) / (a + b),
                        },
                    )
                )
    G.add_edges_from(edges)
    return G


def match(
    rides: pd.DataFrame,
    requests: pd.DataFrame,
    matching_obj: str = "u_veh",
) -> Dict:
    request_indexes = {}
    request_indexes_inv = {}
    for i, index in enumerate(requests.index.values):
        request_indexes[index] = i
        request_indexes_inv[i] = index

    im_indexes = {}
    im_indexes_inv = {}
    for i, index in enumerate(rides.index.values):
        im_indexes[index] = i
        im_indexes_inv[i] = index

    nR = requests.shape[0]

    def add_binary_row(requests):
        ret = np.zeros(nR)
        for i in requests.indexes:
            ret[request_indexes[i]] = 1
        return ret

    rides["row"] = rides.apply(
        add_binary_row, axis=1
    )  # row to be used as constrain in optimization
    m = np.vstack(rides["row"].values).T  # creates a numpy array for the constrains

    rides["index"] = rides.index.copy()

    rides = rides.reset_index(drop=True)

    # optimization
    prob = pulp.LpProblem("Matchingproblem", pulp.LpMinimize)  # problem

    variables = pulp.LpVariable.dicts(
        "r", (i for i in rides.index), cat="Binary"
    )  # decision variables

    cost_col = matching_obj
    if cost_col == "degree":
        costs = rides.indexes.apply(lambda x: -(10 ** len(x)))
    elif cost_col == "u_pax":
        costs = rides[cost_col]  # set the costs
    else:
        costs = rides[cost_col]  # set the costs

    prob += (
        pulp.lpSum([variables[i] * costs[i] for i in variables]),
        "ObjectiveFun",
    )  # ffef

    j = 0  # adding constrains
    for imr in m:
        j += 1
        prob += pulp.lpSum(
            [imr[i] * variables[i] for i in variables if imr[i] > 0]
        ) == 1, "c" + str(j)

    solver = pulp.get_solver(solver_for_pulp())
    solver.msg = False
    prob.solve(solver)

    assert (
        pulp.value(prob.objective) <= sum(costs[:nR]) + 2
    )  # we did not go above original

    locs = {}
    for variable in prob.variables():
        i = int(variable.name.split("_")[1])

        locs[im_indexes_inv[i]] = int(variable.varValue)

    return locs


def solver_for_pulp() -> str:
    system = platform.system()
    if system == "Windows":
        return "GLPK_CMD"
    else:
        return "PULP_CBC_CMD"


def calculate_results(rides: pd.DataFrame, requests: pd.DataFrame) -> float:
    match(rides, requests)
    fin = rides.loc[rides["selected"] == 1]
    return sum(fin["PassSecTrav_ns"]) - sum(fin["u_veh"])


EPS = 1e-15


def just_balance_pool(x, adj, s, mask=None, normalize=True):
    r"""The Just Balance pooling operator from the `"Simplifying Clustering with
    Graph Neural Networks" <https://arxiv.org/abs/2207.08779>`_ paper copied from https://github.com/FilippoMB/Simplifying-Clustering-with-Graph-Neural-Networks/tree/main
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # Loss
    ss = torch.matmul(s.transpose(1, 2), s)
    ss_sqrt = torch.sqrt(ss + EPS)
    loss = torch.mean(-_rank3_trace(ss_sqrt))
    if normalize:
        loss = loss / torch.sqrt(torch.tensor(num_nodes * k))

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum("ijk->ij", out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, loss


def _rank3_trace(x):
    return torch.einsum("ijj->i", x)


def col_to_one_hots(df, col):
    one_hots = pd.get_dummies(df[col], prefix=col)
    df = df.merge(one_hots, left_index=True, right_index=True)
    return df


def get_features(requests: pd.DataFrame) -> torch.tensor:
    def str_datetime_to_float(col):
        col = pd.to_datetime(col)
        min_col = col.min()
        col = (col - min_col).apply(lambda x: x.total_seconds())
        return col

    requests = requests.drop(columns=["id", "pax_id", "status", "VoT", "u_PT"])
    requests = col_to_one_hots(requests, "origin")
    requests = col_to_one_hots(requests, "destination")
    requests = col_to_one_hots(requests, "kind")
    requests = col_to_one_hots(requests, "position")
    requests = col_to_one_hots(requests, "ride_id")
    requests["dropoff_datetime"] = pd.to_datetime(requests["dropoff_datetime"])
    requests["pickup_datetime"] = str_datetime_to_float(requests["pickup_datetime"])
    requests["dropoff_datetime"] = str_datetime_to_float(requests["dropoff_datetime"])
    requests = (requests - requests.min()) / (requests.max() - requests.min())
    return torch.tensor(requests.to_numpy()).float()


def graph_to_undirected(
        directed_graph: nx.DiGraph or nx.Graph,
        relative_utility: bool = False
) -> nx.Graph:
    """
    Create an undirected version of directed graph
    while maintaining weights (comp. to_undirected())
    """
    graph = nx.Graph()
    graph.add_nodes_from(directed_graph.nodes())

    var_name = 'frac_u' if relative_utility else 'u'

    edges_to_add = {}

    for i, j, data in directed_graph.edges(data=True):
        if (j, i) in edges_to_add.keys():
            edges_to_add[(j, i)] += data[var_name]
        else:
            try:
                edges_to_add[(i, j)] += data[var_name]
            except KeyError:
                edges_to_add[(i, j)] = data[var_name]

    for n1, n2 in edges_to_add.keys():
        graph.add_edge(n1, n2, weight=edges_to_add[(n1, n2)])

    return graph