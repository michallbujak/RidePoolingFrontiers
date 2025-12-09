# switzerland_multimodal_stacked.py
# Final clean version: Switzerland multimodal graph (tram, metro, rail, bus + car)
# Nearby stops (≤ 400 m) merged into supernodes → no explicit walking edges
# Fixed: graph_from_bbox tuple format + official GTFS URL (Nov 2025)

import networkx as nx
import gtfs_kit as gk
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import requests
from pathlib import Path
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
import numpy as np

# ------------------------------------------------------------------
# 1. Download GTFS (official daily updated FP2025)
# ------------------------------------------------------------------
GTFS_URL = "https://data.opentransportdata.swiss/dataset/6cca1dfb-e53d-4da8-8d49-4797b3e768e3/resource/ad1c289b-22b5-4043-8d6b-e5ea59706c93/download/gtfs_fp2025_20251120.zip"  # Official, daily updated
gtfs_path = Path("gtfs_fp2025_20251120.zip")

if not gtfs_path.exists():
    print("Downloading Swiss GTFS (~500 MB)...")
    r = requests.get(GTFS_URL, stream=True)
    with open(gtfs_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

print("Loading GTFS feed...")
feed = gk.read_feed(gtfs_path, dist_units="km")

# ------------------------------------------------------------------
# 2. Keep only tram, metro, rail, bus
# ------------------------------------------------------------------
ALLOWED_MODES = {0: "tram", 1: "metro", 2: "rail", 3: "bus"}

valid_route_ids = feed.routes[
    feed.routes["route_type"].isin(ALLOWED_MODES.keys())
]["route_id"]

valid_trip_ids = feed.trips[
    feed.trips["route_id"].isin(valid_route_ids)
]["trip_id"]

stop_times = feed.stop_times[feed.stop_times["trip_id"].isin(valid_trip_ids)].copy()

print(f"Kept {len(valid_route_ids):,} routes and {len(valid_trip_ids):,} trips (tram/metro/rail/bus)")

# ------------------------------------------------------------------
# 3. Prepare stops GeoDataFrame (projected to meters)
# ------------------------------------------------------------------
stops = feed.stops.copy()
gdf = gpd.GeoDataFrame(
    stops,
    geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
    crs="EPSG:4326"
).to_crs("EPSG:3857")  # Web Mercator → accurate distances in meters

# ------------------------------------------------------------------
# 4. Cluster stops into supernodes (max 400 m apart)
# ------------------------------------------------------------------
print("Clustering stops into supernodes (≤ 400 m)...")
coords_m = np.array([[p.x, p.y] for p in gdf.geometry])

db = DBSCAN(eps=400, min_samples=1, metric='euclidean').fit(coords_m)
gdf["supernode"] = db.labels_
n_supernodes = len(set(db.labels_))

print(f"→ Clustered {len(gdf):,} stops into {n_supernodes:,} supernodes")

stop_to_supernode = dict(zip(gdf.stop_id, gdf.supernode))

# Compute centroid of each cluster (with proper CRS)
supernode_coords = gdf.groupby("supernode").geometry.apply(lambda g: g.unary_union.centroid)
supernode_coords = gpd.GeoSeries(supernode_coords, crs=gdf.crs)  # Explicit GeoSeries with CRS
supernode_lonlat = supernode_coords.to_crs("EPSG:4326")

supernode_lat = supernode_lonlat.y
supernode_lon = supernode_lonlat.x

# ------------------------------------------------------------------
# 5. Build NetworkX graph
# ------------------------------------------------------------------
G = nx.MultiDiGraph(name="Switzerland_Stacked_Clean_2025")

# Add supernodes
print("Adding supernodes to graph...")
for sid in gdf.supernode.unique():
    lat = supernode_lat[sid]
    lon = supernode_lon[sid]
    members = gdf[gdf.supernode == sid].stop_id.tolist()
    G.add_node(sid, lat=lat, lon=lon, member_stops=members, mode="hub", pos=(lon, lat))

# Add transit edges between different supernodes
print("Adding scheduled transit edges...")
edges_added = 0
for trip_id, group in stop_times.groupby("trip_id"):
    seq = group.sort_values("stop_sequence")
    stops_in_trip = seq["stop_id"].tolist()
    supernodes_in_trip = [stop_to_supernode[s] for s in stops_in_trip]

    for i in range(len(stops_in_trip) - 1):
        u_super = supernodes_in_trip[i]
        v_super = supernodes_in_trip[i + 1]
        if u_super == v_super:
            continue  # same hub → free transfer

        u_stop = stops_in_trip[i]
        v_stop = stops_in_trip[i + 1]

        # Travel time
        dep = seq[seq.stop_id == u_stop].iloc[0]["departure_time"]
        arr = seq[seq.stop_id == v_stop].iloc[0]["arrival_time"]
        travel_time = 120
        if pd.notna(dep) and pd.notna(arr):
            try:
                t1 = sum(int(x)*60**i for i,x in enumerate(reversed(dep.split(":"))))
                t2 = sum(int(x)*60**i for i,x in enumerate(reversed(arr.split(":"))))
                travel_time = max(t2 - t1, 30)
            except:
                pass

        # Distance
        u_c = (stops.loc[u_stop, "stop_lat"], stops.loc[u_stop, "stop_lon"])
        v_c = (stops.loc[v_stop, "stop_lat"], stops.loc[v_stop, "stop_lon"])
        distance_km = geodesic(u_c, v_c).km

        # Mode (safe lookup with fallback)
        trips_row = feed.trips.get(trip_id, pd.Series({"route_id": None}))
        route_id = trips_row["route_id"] if trips_row is not None else None
        routes_row = feed.routes.get(route_id, pd.Series({"route_type": 3})) if route_id else pd.Series({"route_type": 3})
        route_type = int(routes_row["route_type"])
        mode = ALLOWED_MODES.get(route_type, "bus")

        G.add_edge(
            u_super, v_super,
            mode=mode,
            trip_id=trip_id,
            travel_time=travel_time,
            distance=distance_km,
            type="scheduled"
        )
        edges_added += 1

print(f"Added {edges_added:,} scheduled transit edges")

# ------------------------------------------------------------------
# 6. Add OSM road network (FIXED: bbox as tuple)
# ------------------------------------------------------------------
print("Downloading OSM road network (Switzerland)...")
# bbox tuple: (north, south, east, west) — full CH with buffer
G_roads = ox.graph_from_bbox((47.908, 45.718, 10.592, 5.856), network_type="all")
# For testing: G_roads = ox.graph_from_place("Zürich, Switzerland", network_type="all")

print(f"OSM: {G_roads.number_of_nodes():,} nodes, {G_roads.number_of_edges():,} edges")

for u, v, k, data in G_roads.edges(keys=True, data=True):
    u_new = f"osm_{u}"
    v_new = f"osm_{v}"
    if u_new not in G:
        G.add_node(u_new, **G_roads.nodes[u], mode="road")
    if v_new not in G:
        G.add_node(v_new, **G_roads.nodes[v], mode="road")
    edge_data = data.copy()
    highway = edge_data.get("highway", "")
    edge_data["mode"] = "car" if isinstance(highway, str) and any(h in highway for h in ["motorway","primary","secondary","tertiary","residential"]) else "walk_bike"
    G.add_edge(u_new, v_new, key=k, **edge_data)

print("Road network merged")

# ------------------------------------------------------------------
# 7. Connect supernodes to road network
# ------------------------------------------------------------------
print("Connecting supernodes to nearest road node...")
access = 0
for sid in list(G.nodes):
    if G.nodes[sid].get("mode") != "hub":
        continue
    lat, lon = G.nodes[sid]["lat"], G.nodes[sid]["lon"]
    nearest = ox.distance.nearest_nodes(G_roads, lon, lat)
    road_node = f"osm_{nearest}"
    dist_m = ox.distance.great_circle_vec(lat, lon, G_roads.nodes[nearest]["y"], G_roads.nodes[nearest]["x"])
    if dist_m <= 500:
        walk_time = int(dist_m / 1.4)
        G.add_edge(sid, road_node, mode="walk", travel_time=walk_time, distance=round(dist_m), type="access")
        G.add_edge(road_node, sid, mode="walk", travel_time=walk_time, distance=round(dist_m), type="egress")
        access += 2

print(f"Added {access:,} access/egress links")

# ------------------------------------------------------------------
# 8. Save
# ------------------------------------------------------------------
print("\nGraph complete!")
print(f"Total nodes: {G.number_of_nodes():,}")
print(f"Total edges: {G.number_of_edges():,}")

nx.write_graphml(G, "switzerland_stacked_clean.graphml")
print("Saved as: switzerland_stacked_clean.graphml")