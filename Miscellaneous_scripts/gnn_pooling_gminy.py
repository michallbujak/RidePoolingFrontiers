import pandas as pd
import numpy as np
import networkx as nx

df = pd.read_csv('data/rejony.att', skiprows=85, delimiter=";", index_col=0)
df = df[[isinstance(t, str) for t in df["WOJEWODZTWO"]]]
df["REGION_ID"] = list(df.index)
df = df.rename(columns={"NAME": "NEIGHBOURS"})
df["NEIGHBOURS"] = df["NEIGHBOURS"].apply(lambda x: [int(t) for t in x.split(',')])
df["NEIGHBOURS"] = df.apply(lambda x: [t for t in x["NEIGHBOURS"] if t != x["REGION_ID"]], axis=1)

adj_graph = nx.Graph()
adj_graph.add_nodes_from([
    (t[1]["REGION_ID"], {"CITIZENS": int(t[1]["MR_LUDN"])})
     for t in df.iterrows()
])

edges = set()
for node, neighbours in df["NEIGHBOURS"].items():
    for neighbour in neighbours:
        edges.add((node, neighbour))



x = 0
