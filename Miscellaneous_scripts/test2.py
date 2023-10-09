import osmnx as ox
import numpy as np
import pandas as pd
import networkx as nx
from random import sample
from datetime import timedelta, datetime
from dotmap import DotMap
from tqdm import tqdm
import time


def translate_to_osmnx(
        raw_data: pd.DataFrame,
        city_graph: nx.MultiDiGraph,
        location_dictionary: dict or None = None,
        **kwargs
) -> (pd.DataFrame, dict):
    """
    Amend an input dataset: convert dataframe with nodes'
    longitudes and latitudes to the osmnx ids
    @param raw_data: dataframe with long, lat
    @param city_graph: graph of the city
    @param location_dictionary: dictionary with locations
    @param kwargs: names of the columns with longs and lats
    @return: .csv with origin and destinations in osmnx format
    """
    origin_long = kwargs.get("origin_long", "ORIGIN_BLOCK_LONGITUDE")
    origin_lat = kwargs.get("origin_lat", "ORIGIN_BLOCK_LATITUDE")
    destination_long = kwargs.get("destination_long", "DESTINATION_BLOCK_LONG")
    destination_lat = kwargs.get("destination_lat", "DESTINATION_BLOCK_LAT")

    df = raw_data.copy()

    if location_dictionary is None:
        location_dictionary = dict()

    locations_list = zip(list(df[origin_long]) + list(df[destination_long]),
                         list(df[origin_lat]) + list(df[destination_lat]))

    for location in locations_list:
        if location in location_dictionary.keys():
            pass
        else:
            location_dictionary[location] = ox.nearest_nodes(
                city_graph,
                float(location[0]),
                float(location[1])
            )

    df["origin"] = df.apply(lambda x: location_dictionary[(x[origin_long], x[origin_lat])],
                            axis=1)

    df["destination"] = df.apply(lambda x: location_dictionary[(x[destination_long], x[destination_lat])],
                                 axis=1)

    return df, location_dictionary


def add_noise_to_data(
        requests: pd.DataFrame,
        skim: pd.DataFrame,
        distance: float or int = 300,
        **kwargs
) -> pd.DataFrame:
    """
    Sample nearby nodes for given centres of areas and sample
    time for the intervals
    @param requests: requests
    @param skim: dataframe with distances (indexes and columns
    are osmnx nodes
    @param distance: distance from the centre point (in metres)
    @param kwargs: time specifics (hours, minutes, seconds)
    @return: amended dataframe with the additional noise
    """
    requests['initial_origin'] = requests['origin']
    requests['initial_destination'] = requests['destination']
    node_dict = {}

    skim.index = [int(t) for t in skim.index]
    skim.columns = [int(t) for t in skim.columns]

    node_list = list(requests['origin']) + list(requests['destination'])

    pbar = tqdm(total=len(node_list))

    for node in node_list:
        df_node = skim.loc[skim[node] < distance]
        pbar.update(1)
        if df_node.empty:
            node_dict[node] = [node]
        else:
            node_dict[node] = list(df_node.index)

    df['origin'] = df['origin'].apply(lambda x: sample(node_dict[x], 1)[0])
    df['destination'] = df['destination'].apply(lambda x: sample(node_dict[x], 1)[0])

    if kwargs.get('hours', 0) > 0:
        kwargs['minutes'] = 60
        kwargs['seconds'] = 60

    if kwargs.get('minutes', 0) > 0:
        kwargs['seconds'] = 60

    requests['pickup_datetime'] = \
        requests['pickup_datetime'].apply(lambda x: x +
                                                    timedelta(hours=np.random.randint(0, kwargs.get('hours', 0) + 1)))
    requests['pickup_datetime'] = \
        requests['pickup_datetime'].apply(lambda x: x +
                                                    timedelta(
                                                        minutes=np.random.randint(0, kwargs.get('minutes', 0) + 1)))
    requests['pickup_datetime'] = \
        requests['pickup_datetime'].apply(lambda x: x +
                                                    timedelta(
                                                        seconds=np.random.randint(0, kwargs.get('seconds', 0) + 1)))

    return requests.sort_values("pickup_datetime")


def download_G(inData, _params, make_skims=True):
    # uses osmnx to download the graph
    inData.G = ox.graph_from_place(_params.city, network_type='drive')
    inData.nodes = pd.DataFrame.from_dict(dict(inData.G.nodes(data=True)), orient='index')
    if make_skims:
        inData.skim_generator = nx.all_pairs_dijkstra_path_length(inData.G,
                                                                  weight='length',
                                                                  cutoff=50000)
        inData.skim_dict = dict(inData.skim_generator)  # filled dict is more usable
        inData.skim = pd.DataFrame(inData.skim_dict).fillna(_params.dist_threshold).T.astype(
            int)  # and dataframe is more intuitive
    return inData


def save_G(inData, _params, path=None):
    # saves graph and skims to files
    ox.save_graphml(inData.G, filepath=_params.paths.G)
    inData.skim.to_csv(_params.paths.skim, chunksize=20000000)


params = DotMap()
params.city = "Chicago, USA"
params.dist_threshold = 10000
params.paths.G = r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\ExMAS\data\graphs\Chicago.graphml"
params.paths.skim = r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\ExMAS\data\graphs\Chicago2.csv"

# dm = DotMap()
#
# dm = download_G(dm, params)
# save_G(dm, params)


_file_name = "\chicago_2023"
df = pd.read_csv("C:\\Users\\szmat\\Documents\\GitHub\\ExMAS_sideline\\ExMAS\\data\\{0}.csv".format(_file_name),
                 index_col="Unnamed: 0")
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df = df.loc[(df["pickup_datetime"] >= datetime(2023, 2, 1)) & (df["pickup_datetime"] <= datetime(2023, 2, 28))]


# for _name in ["Pickup Centroid Latitude", "Pickup Centroid Longitude",
#               "Dropoff Centroid Longitude", "Dropoff Centroid Latitude"]:
#     df = df.loc[[not np.isnan(t) for t in df[_name]]]
#
# df, loc_dict = translate_to_osmnx(
#     raw_data=df,
#     city_graph=ox.load_graphml(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\ExMAS\data\graphs\Chicago.graphml"),
#     location_dictionary=None,
#     origin_lat="Pickup Centroid Latitude",
#     origin_long="Pickup Centroid Longitude",
#     destination_long="Dropoff Centroid Longitude",
#     destination_lat="Dropoff Centroid Latitude"
# )
#
# df["pickup_datetime"] = pd.to_datetime(df["Trip Start Timestamp"])
#
# df.to_csv("C:\\Users\\szmat\\Documents\\GitHub\\ExMAS_sideline\\ExMAS\\data\\{0}.csv".format(_file_name))
print("------------------DONE--------------------")

# nodes = np.unique(list(df['origin']) + list(df['destination']))
# mini_skim = np.random.randint(0, 10000, (len(nodes), len(nodes)))
# mini_skim = (mini_skim + mini_skim.T) / 2
# np.fill_diagonal(mini_skim, 0)
# mini_skim = pd.DataFrame(mini_skim)
# mini_skim.index = nodes
# mini_skim.columns = nodes

skim = pd.read_csv(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\ExMAS\data\graphs\Chicago.csv",
                   index_col="Unnamed: 0")
# df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df = add_noise_to_data(df, skim, minutes=15)
df.to_csv("C:\\Users\\szmat\\Documents\\GitHub\\ExMAS_sideline\\ExMAS\\data\\{0}_amended_feb.csv".format(_file_name))
