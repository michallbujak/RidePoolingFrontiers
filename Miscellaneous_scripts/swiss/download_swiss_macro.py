# switzerland_sbb_red_FINAL_CLEAN.py
# FINAL – Flexible matching → canonical cities → dense red SBB rail

import networkx as nx
import gtfs_kit as gk
import pandas as pd
import requests
from pathlib import Path
from geopy.distance import geodesic
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# 1. GTFS
# --------------------------------------------------------------
GTFS_URL = "https://data.opentransportdata.swiss/dataset/6cca1dfb-e53d-4da8-8d49-4797b3e768e3/resource/ad1c289b-22b5-4043-8d6b-e5ea59706c93/download/gtfs_fp2025_20251120.zip"
gtfs_path = Path("gtfs_fp2025_20251120.zip")

if not gtfs_path.exists():
    print("Downloading GTFS...")
    r = requests.get(GTFS_URL, stream=True)
    with open(gtfs_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

feed = gk.read_feed(gtfs_path, dist_units="km")
stops = feed.stops.copy()
routes = feed.routes
trips = feed.trips
stop_times = feed.stop_times

# --------------------------------------------------------------
# 2. Canonical city list + keywords
# --------------------------------------------------------------
canonical_cities = {
    "Zürich": ["Zürich", "Zuerich", "Zürich HB", "Zürich Oerlikon", "Zürich Altstetten", "Zürich Stadelhofen",
               "Zürich Flughafen", "Zürich Hauptbahnhof"],
    "Genève": ["Genève", "Geneve", "Genf"],
    "Basel": ["Basel", "Basle", "Bâle", "Basel SBB", "Basel Bad Bf"],
    "Bern": ["Bern", "Berne"],
    "Lausanne": ["Lausanne", "Renens"],
    "Winterthur": ["Winterthur"],
    "Luzern": ["Luzern", "Lucerne"],
    "St. Gallen": ["St. Gallen", "Sankt Gallen"],
    "Lugano": ["Lugano"],
    "Biel/Bienne": ["Biel", "Bienne"],
    "Thun": ["Thun"],
    "Olten": ["Olten"],
    "Aarau": ["Aarau"],
    "Solothurn": ["Solothurn"],
    "Chur": ["Chur", "Coire"],
    "Bellinzona": ["Bellinzona"],
    "Locarno": ["Locarno"],
    "Fribourg": ["Fribourg", "Freiburg", "Freiburg im Breisgau"],
    "Neuchâtel": ["Neuchâtel", "Neuchatel"],
    "Sion": ["Sion", "Sitten"],
    "Brig": ["Brig"],
    "Schaffhausen": ["Schaffhausen"],
    "Zug": ["Zug"],
    "Interlaken": ["Interlaken"],
    "Spiez": ["Spiez"],
    "Visp": ["Visp"],
    "Montreux": ["Montreux"],
    "Vevey": ["Vevey"],
    "Yverdon-les-Bains": ["Yverdon"],
    "Delémont": ["Delémont"],
    "Pfäffikon SZ": ["Pfäffikon", "Pfaeffikon"],
    "Arth-Goldau": ["Arth-Goldau"],
    "Sargans": ["Sargans"],
    "Rapperswil": ["Rapperswil"]
}

# Build reverse map: stop name → canonical city
stop_to_city = {}
for city, keywords in canonical_cities.items():
    for kw in keywords:
        stop_to_city[kw.lower()] = city


def assign_city(stop_name):
    name_lower = stop_name.lower()
    for kw, city in [(k.lower(), c) for k, c in stop_to_city.items()]:
        if kw in name_lower:
            return city
    return None


stops["canonical_city"] = stops["stop_name"].apply(assign_city)
selected_stops = stops[stops["canonical_city"].notna()].copy()

print(f"{len(selected_stops)} stops matched to {len(selected_stops['canonical_city'].unique())} canonical cities")

# --------------------------------------------------------------
# 3. Build graph (one node per canonical city)
# --------------------------------------------------------------
G = nx.Graph()

for city in selected_stops["canonical_city"].unique():
    city_stops = selected_stops[selected_stops["canonical_city"] == city]
    # Use average coordinates
    lon = city_stops["stop_lon"].mean()
    lat = city_stops["stop_lat"].mean()
    G.add_node(city, x=lon, y=lat)

# --------------------------------------------------------------
# 4. RAIL: Consecutive canonical cities in trips
# --------------------------------------------------------------
print("Adding rail connections...")
rail_edges = set()

# Include all SBB rail types
sbb_route_types = [2, 100, 101, 102, 103, 104, 105]
sbb_trips = trips[trips["route_id"].isin(routes[routes["route_type"].isin(sbb_route_types)]["route_id"])]["trip_id"]

for trip_id in sbb_trips:
    seq = stop_times[stop_times["trip_id"] == trip_id].sort_values("stop_sequence")
    major_stops = seq[seq["stop_id"].isin(selected_stops["stop_id"])].copy()
    if len(major_stops) < 2: continue

    major_stops = major_stops.merge(selected_stops[["stop_id", "canonical_city"]], on="stop_id")
    cities = major_stops["canonical_city"].tolist()

    for i in range(len(cities) - 1):
        a, b = sorted([cities[i], cities[i + 1]])
        if a != b:
            rail_edges.add((a, b))

for a, b in rail_edges:
    G.add_edge(a, b, mode="rail")

print(f"{len(rail_edges)} rail edges")

# --------------------------------------------------------------
# 5. ROAD: Euclidean < 120 km
# --------------------------------------------------------------
print("Adding road connections...")
coords = {c: (G.nodes[c]["y"], G.nodes[c]["x"]) for c in G.nodes}
for i, u in enumerate(G.nodes):
    for j in range(i + 1, len(G.nodes)):
        v = list(G.nodes)[j]
        if geodesic(coords[u], coords[v]).km <= 120:
            G.add_edge(u, v, mode="road")

# --------------------------------------------------------------
# 6. PLOT – Thick red rail
# --------------------------------------------------------------
plt.figure(figsize=(18, 14))
pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}

road_edges = [(u, v) for u, v, d in G.edges(data=True) if d["mode"] == "road"]
rail_edges = [(u, v) for u, v, d in G.edges(data=True) if d["mode"] == "rail"]

nx.draw_networkx_edges(G, pos, edgelist=road_edges, width=0.6, alpha=0.3, edge_color="gray")
nx.draw_networkx_edges(G, pos, edgelist=rail_edges, width=4, alpha=0.95, edge_color="#d62728")
nx.draw_networkx_nodes(G, pos, node_size=120, node_color="#d62728", edgecolors="black")
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

plt.title("Switzerland – SBB Rail Network (Red) + Major Roads", fontsize=18)
plt.axis("off")
plt.tight_layout()
plt.savefig("switzerland_sbb_red_final.png", dpi=400, bbox_inches="tight", facecolor="white")
print("SAVED: switzerland_sbb_red_final.png")

nx.write_gexf(G, "switzerland_sbb_red_final.gexf")
print("SAVED: switzerland_sbb_red_final.gexf")

print(f"DONE! {G.number_of_nodes()} cities, {G.number_of_edges()} edges")