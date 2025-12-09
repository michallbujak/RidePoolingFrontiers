#!/usr/bin/env python3
"""
swiss_gtfs_hubs_graph.py

- Downloads GTFS zip (URL provided by user)
- Reads stops, trips, stop_times
- Clusters stops into "city/hub" nodes (DBSCAN on projected coordinates)
- Selects top clusters (by size / route-count) to obtain between 30 and 100 nodes,
  prioritizing major rail hubs
- Builds a NetworkX MultiGraph:
    - 'rail' edges: from GTFS consecutive stops mapped to cluster nodes (no transitive links)
    - 'road' edges: straight-line (geodesic) connections between nearest neighbor clusters
- Saves GraphML and a simple plot (roads thin, rails thicker)
"""

import os
import io
import zipfile
import math
import requests
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from sklearn.cluster import DBSCAN
from pyproj import CRS, Transformer

# ----------------- USER PARAMETERS -----------------
GTFS_URL = "https://data.opentransportdata.swiss/dataset/6cca1dfb-e53d-4da8-8d49-4797b3e768e3/resource/ad1c289b-22b5-4043-8d6b-e5ea59706c93/download/gtfs_fp2025_20251120.zip"
OUTPUT_DIR = Path("../output")
OUTPUT_DIR.mkdir(exist_ok=True)
GRAPHML_OUT = OUTPUT_DIR / "combined_graph.graphml"
PLOT_OUT = OUTPUT_DIR / "combined_graph.png"

# clustering & selection parameters
DBSCAN_EPS_METERS = 3000    # cluster radius in meters (3 km) — adjusts how stops group into a 'city/hub'
DBSCAN_MIN_SAMPLES = 1      # allow small clusters; we'll select only the larger ones later
TARGET_MIN_NODES = 30
TARGET_MAX_NODES = 100

# road edges: number of nearest neighbor nodes to consider for adding a 'road' link
ROAD_K_NEIGHBORS = 3

# For better repeatability, set random seed where applicable (not strictly needed for DBSCAN)
# ----------------------------------------------------

def download_gtfs(gtfs_url: str) -> zipfile.ZipFile:
    print("Downloading GTFS from:", gtfs_url)
    r = requests.get(gtfs_url, stream=True, timeout=60)
    r.raise_for_status()
    data = r.content
    z = zipfile.ZipFile(io.BytesIO(data))
    print("Downloaded GTFS zip, contents:", z.namelist()[:10])
    return z

def read_gtfs_from_zip(z: zipfile.ZipFile):
    # required files
    required = ["stops.txt", "trips.txt", "stop_times.txt", "routes.txt"]
    members = set(z.namelist())
    missing = [f for f in required if f not in members]
    if missing:
        raise RuntimeError(f"GTFS zip missing required files: {missing}")

    def read_csv(name):
        with z.open(name) as f:
            return pd.read_csv(f, dtype=str)

    stops = read_csv("stops.txt")
    trips = read_csv("trips.txt")
    stop_times = read_csv("stop_times.txt")
    routes = read_csv("routes.txt")
    # convert relevant numeric columns
    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)
    # keep stops with coordinates
    stops = stops.dropna(subset=["stop_lat", "stop_lon"]).reset_index(drop=True)
    return stops, trips, stop_times, routes

def stops_to_projected_coords(stops_df: pd.DataFrame, target_epsg=3857):
    # use pyproj to transform lat/lon to a projected coordinate system — EPSG:3857 is fine for clustering
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
    xs, ys = transformer.transform(stops_df["stop_lon"].values, stops_df["stop_lat"].values)
    stops_df = stops_df.copy()
    stops_df["x"] = xs
    stops_df["y"] = ys
    return stops_df

def cluster_stops_dbscan(stops_proj: pd.DataFrame, eps_m=DBSCAN_EPS_METERS, min_samples=DBSCAN_MIN_SAMPLES):
    coords = stops_proj[["x", "y"]].values
    db = DBSCAN(eps=eps_m, min_samples=min_samples, metric="euclidean").fit(coords)
    labels = db.labels_
    stops_proj = stops_proj.copy()
    stops_proj["cluster"] = labels
    # ignore noise (-1) only if occurs; we keep them as their own cluster if -1
    return stops_proj

def compute_cluster_stats(stops_proj: pd.DataFrame, stop_times: pd.DataFrame, trips: pd.DataFrame):
    # Map stop_id -> number of distinct routes / trips that touch it (a proxy for hub importance)
    # We'll compute distinct route_ids from trips via stop_times->trip_id->route_id
    print("Computing route counts per stop...")
    # ensure types
    st = stop_times[["trip_id", "stop_id"]].dropna()
    tr = trips[["trip_id", "route_id"]].dropna()
    merged = st.merge(tr, on="trip_id", how="left")
    # route counts per stop
    route_counts = merged.groupby("stop_id")["route_id"].nunique().fillna(0).astype(int).to_dict()
    stops_proj = stops_proj.copy()
    stops_proj["route_count"] = stops_proj["stop_id"].map(route_counts).fillna(0).astype(int)

    # cluster aggregates: size, distinct stops, total route_count, centroid
    cluster_groups = []
    for cid, grp in stops_proj.groupby("cluster"):
        centroid_x = grp["x"].mean()
        centroid_y = grp["y"].mean()
        centroid_lat = grp["stop_lat"].astype(float).mean()
        centroid_lon = grp["stop_lon"].astype(float).mean()
        cluster_groups.append({
            "cluster": int(cid),
            "n_stops": len(grp),
            "n_unique_stop_ids": grp["stop_id"].nunique(),
            "sum_route_count": int(grp["route_count"].sum()),
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "sample_stop_names": list(grp["stop_name"].unique())[:6],
        })
    clusters_df = pd.DataFrame(cluster_groups).sort_values(["sum_route_count", "n_stops"], ascending=False).reset_index(drop=True)
    return stops_proj, clusters_df

def select_clusters_balanced(clusters_df: pd.DataFrame, min_nodes=TARGET_MIN_NODES, max_nodes=TARGET_MAX_NODES):
    # Select clusters in order of importance (sum_route_count then n_stops) until we reach >=min_nodes,
    # but do not exceed max_nodes. If there are fewer clusters than min_nodes, we'll take them all.
    if len(clusters_df) <= max_nodes:
        take = len(clusters_df)
    else:
        # pick top by sum_route_count primarily
        take = min(max_nodes, max(min_nodes, (clusters_df["sum_route_count"] > 0).sum()))
        # If that yields too few, increase
        if take < min_nodes:
            take = min(max_nodes, min_nodes)
    selected = clusters_df.head(take).copy()
    print(f"Selecting {len(selected)} clusters (nodes) for the graph.")
    return selected

def build_nodes_from_clusters(selected_clusters: pd.DataFrame):
    nodes = {}
    for idx, row in selected_clusters.iterrows():
        nid = int(row["cluster"])
        nodes[nid] = {
            "cluster": nid,
            "n_stops": int(row["n_stops"]),
            "sum_route_count": int(row["sum_route_count"]),
            "centroid_x": float(row["centroid_x"]),
            "centroid_y": float(row["centroid_y"]),
            "lat": float(row["centroid_lat"]),
            "lon": float(row["centroid_lon"]),
            "sample_names": row["sample_stop_names"],
        }
    return nodes

def find_stop_cluster_map(stops_proj: pd.DataFrame, selected_cluster_ids):
    # map stop_id -> cluster if its cluster in selected set, else map to nearest selected cluster by distance
    print("Mapping every GTFS stop to a selected cluster (nearest if its cluster wasn't selected)...")
    # build selected cluster centroids
    selected = stops_proj[stops_proj["cluster"].isin(selected_cluster_ids)].groupby("cluster").agg({
        "x":"mean", "y":"mean"
    }).reset_index()
    centroids = selected.set_index("cluster")[["x","y"]].to_dict(orient="index")

    def nearest_cluster_for_point(x,y):
        best = None
        best_d = float("inf")
        for cid, c in centroids.items():
            dx = x - c["x"]
            dy = y - c["y"]
            d = dx*dx + dy*dy
            if d < best_d:
                best_d = d
                best = cid
        return int(best)

    stop_to_cluster = {}
    for _, row in stops_proj.iterrows():
        sid = row["stop_id"]
        cid_orig = int(row["cluster"])
        if cid_orig in centroids:
            stop_to_cluster[sid] = cid_orig
        else:
            stop_to_cluster[sid] = nearest_cluster_for_point(row["x"], row["y"])
    return stop_to_cluster

def build_rail_edges_from_gtfs(stop_times: pd.DataFrame, trips: pd.DataFrame, stop_to_cluster: dict):
    # We'll iterate over trips in stop_times ordered by trip_id and stop_sequence,
    # and for each consecutive pair of stops (A->B) add an edge between their clusters if both clusters exist and differ.
    print("Building rail edges from GTFS consecutive stops (no transitive links)...")
    st = stop_times[["trip_id", "stop_sequence", "stop_id"]].copy()
    # ensure numeric sorting
    st["stop_sequence"] = st["stop_sequence"].astype(float)
    st = st.sort_values(["trip_id", "stop_sequence"])
    rail_edges = Counter()
    # For use later we can also keep an example trip_id per edge
    example_trip = {}
    prev_trip = None
    prev_stop = None
    for _, row in st.iterrows():
        trip_id = row["trip_id"]
        stop_id = row["stop_id"]
        if trip_id != prev_trip:
            prev_trip = trip_id
            prev_stop = stop_id
            continue
        a = stop_to_cluster.get(prev_stop)
        b = stop_to_cluster.get(stop_id)
        if a is None or b is None:
            prev_stop = stop_id
            continue
        if a != b:
            key = (int(a), int(b))
            rail_edges[key] += 1
            # store an example trip id
            if key not in example_trip:
                example_trip[key] = trip_id
        prev_stop = stop_id
    # convert to list of edges with weight (# of times seen)
    edges = []
    for (a,b), w in rail_edges.items():
        edges.append({"u":int(a), "v":int(b), "weight":int(w), "trip_example": example_trip.get((a,b))})
    print(f"Built {len(edges)} directed rail edge types from GTFS.")
    return edges

def haversine_meters(lat1, lon1, lat2, lon2):
    # returns distance in meters between two lat/lon pairs
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def build_road_edges_knn(nodes: dict, k=ROAD_K_NEIGHBORS):
    # naive O(N^2) k-NN for small N (<=100) is fine. Add undirected road edges between node and its k nearest neighbors.
    print("Building approximate 'road' (straight-line) edges using nearest neighbors...")
    node_items = list(nodes.items())
    coords = []
    ids = []
    for nid, data in node_items:
        coords.append((data["lat"], data["lon"]))
        ids.append(nid)
    road_edges = []
    for i, nid in enumerate(ids):
        lat1, lon1 = coords[i]
        dists = []
        for j, nid2 in enumerate(ids):
            if nid == nid2: continue
            lat2, lon2 = coords[j]
            d = haversine_meters(lat1, lon1, lat2, lon2)
            dists.append((d, nid2, lat2, lon2))
        dists.sort()
        # connect to k nearest neighbors (undirected)
        for d, nid2, la, lo in dists[:k]:
            # add both directions? we'll add as undirected edge (store once with sorted node ids)
            a, b = sorted([nid, nid2])
            road_edges.append({"u":int(a), "v":int(b), "distance_m":float(d)})
    # deduplicate
    unique = {}
    for e in road_edges:
        key = (e["u"], e["v"])
        if key not in unique or unique[key]["distance_m"] > e["distance_m"]:
            unique[key] = e
    edges = list(unique.values())
    print(f"Built {len(edges)} undirected road edges (approx).")
    return edges

def build_graph(nodes: dict, rail_edges: list, road_edges: list):
    print("Assembling NetworkX MultiGraph...")
    G = nx.MultiGraph()
    # add nodes
    for nid, data in nodes.items():
        G.add_node(nid, **data)
    # add rail edges (directed sense — but we'll store in undirected MultiGraph with attribute 'rail' and direction info)
    for e in rail_edges:
        u, v = int(e["u"]), int(e["v"])
        # add as an edge — for MultiGraph we can add directed semantics with attribute 'directed': True and 'from'/'to'
        G.add_edge(u, v, key=f"rail_{u}_{v}", mode="rail", weight=e.get("weight",1), trip_example=e.get("trip_example"))
    # add road edges (undirected)
    for e in road_edges:
        u, v = int(e["u"]), int(e["v"])
        dist = float(e["distance_m"])
        G.add_edge(u, v, key=f"road_{u}_{v}", mode="road", distance_m=dist)
    print(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    return G

def save_graph_and_plot(G: nx.MultiGraph, nodes: dict, out_graphml: Path, out_plot: Path):
    print("Saving GraphML to", out_graphml)
    # NetworkX write_graphml has issues with non-string keys; convert attributes to serializable
    H = nx.MultiGraph()
    for n, attr in G.nodes(data=True):
        H.add_node(n, **{k: (v if isinstance(v,(str,int,float,bool)) else str(v)) for k,v in attr.items()})
    for u,v,k,data in G.edges(keys=True, data=True):
        H.add_edge(u, v, **{k2:(v2 if isinstance(v2,(str,int,float,bool)) else str(v2)) for k2,v2 in data.items()})
    nx.write_graphml(H, str(out_graphml))
    print("GraphML written.")

    # simple plot
    print("Plotting graph to", out_plot)
    plt.figure(figsize=(12,10))
    # prepare node positions in lon/lat projected to screen by simple scatter
    xs = []
    ys = []
    labels = []
    for n, data in nodes.items():
        xs.append(data["lon"])
        ys.append(data["lat"])
        labels.append(str(n))
    # draw road edges thin gray
    for u,v,data in G.edges(data=True):
        if data.get("mode") == "road":
            x1 = nodes[u]["lon"]; y1 = nodes[u]["lat"]
            x2 = nodes[v]["lon"]; y2 = nodes[v]["lat"]
            plt.plot([x1,x2],[y1,y2], linewidth=0.6, alpha=0.6, zorder=1)
    # draw rail edges thicker colored
    for u,v,data in G.edges(data=True):
        if data.get("mode") == "rail":
            x1 = nodes[u]["lon"]; y1 = nodes[u]["lat"]
            x2 = nodes[v]["lon"]; y2 = nodes[v]["lat"]
            plt.plot([x1,x2],[y1,y2], linewidth=1.4, alpha=0.9, zorder=2, color="#d62728")
    # nodes
    plt.scatter(xs, ys, s=40, zorder=3, edgecolor="k", linewidth=0.3)
    # annotate a few node labels (only the top k by sum_route_count)
    node_ranks = sorted(nodes.items(), key=lambda kv: kv[1]["sum_route_count"], reverse=True)
    for n, data in node_ranks[:30]:
        plt.text(data["lon"], data["lat"], str(n), fontsize=8, verticalalignment="bottom", horizontalalignment="center")
    plt.title("Combined network (rail edges from GTFS; road edges = nearest-neighbor straight lines)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(str(out_plot), dpi=200)
    plt.close()
    print("Plot saved.")

def main():
    z = download_gtfs(GTFS_URL)
    stops, trips, stop_times, routes = read_gtfs_from_zip(z)
    print(f"GTFS stops: {len(stops)}, trips: {len(trips)}, stop_times rows: {len(stop_times)}")

    stops_proj = stops_to_projected_coords(stops, target_epsg=3857)
    stops_proj = cluster_stops_dbscan(stops_proj, eps_m=DBSCAN_EPS_METERS, min_samples=DBSCAN_MIN_SAMPLES)

    stops_proj, clusters_df = compute_cluster_stats(stops_proj, stop_times, trips)
    print("Top clusters (preview):")
    print(clusters_df.head(10).to_string(index=False))

    selected_clusters = select_clusters_balanced(clusters_df, min_nodes=TARGET_MIN_NODES, max_nodes=TARGET_MAX_NODES)
    selected_cluster_ids = set(selected_clusters["cluster"].tolist())

    nodes_dict = build_nodes_from_clusters(selected_clusters)
    stop_to_cluster = find_stop_cluster_map(stops_proj, selected_cluster_ids)

    rail_edges = build_rail_edges_from_gtfs(stop_times, trips, stop_to_cluster)
    road_edges = build_road_edges_knn(nodes_dict, k=ROAD_K_NEIGHBORS)

    G = build_graph(nodes_dict, rail_edges, road_edges)

    save_graph_and_plot(G, nodes_dict, GRAPHML_OUT, PLOT_OUT)
    print("All done. Outputs in", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()
