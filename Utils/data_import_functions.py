import sys
import os
import random
import math
import logging
from datetime import timedelta, datetime
import secrets

# side packages
import pandas as pd
import osmnx as ox
import networkx as nx
import numpy as np
import dotmap
from tqdm import tqdm


def download_graph_skim(
        city: str,
        max_dist: float or int = 10000
):
    """
    Download city graph to obtain skim matrix
    :param city: name of the city
    :param max_dist: max distance to fill blanks
    :return:
    """
    city_graph = ox.graph_from_place(city, network_type='drive')
    skim_generator = nx.all_pairs_dijkstra_path_length(city_graph, weight='length')
    skim_dict = dict(skim_generator)  # filled dict is more usable
    skim_matrix = pd.DataFrame(skim_dict).fillna(max_dist).T.astype(int)
    # and dataframe is more intuitive
    return city_graph, skim_matrix


def amend_demand_structure(
        requests: pd.DataFrame,
        skim_matrix: pd.DataFrame,
        logger: logging.Logger or None = None,
        **kwargs
) -> (dotmap.DotMap, dotmap.DotMap):
    """
    Function designed to translate a dataframe with origins, destinations and times
    to the suitable ExMAS format
    ---
    @param requests: dataframe with columns including origin, destination and
    request time. If non-standard column names, refer to kwargs
    @param params: configuration
    @param logger: if you want to log the activities, pass a logger
    @param kwargs: special names for columns: origin_name, destination_name,
     pickup_datetime_name
    @return: inputs for the ExMAS main()
    ---
    Example
    ---
    params_nyc = get_config(PATH)
    requests = pd.read_csv(params_nyc.paths.nyc_requests)
    dotmap_data, params_nyc = ExMAS.utils.create_input_dotmap(requests, params_nyc)
    results = ExMAS.main(dotmap_data, params_nyc, False)
    """
    if 'origin' in requests.columns and 'destination' in requests.columns:
        pass
    else:
        requests['origin'] = kwargs.get('origin_name', 'origin')
        requests['destination'] = kwargs.get('destination_name', 'destination')

    requests['status'] = 0
    requests['dist'] = requests.apply(lambda request:
                                      skim_matrix.loc[request['origin'], request['destination']],
                                      axis=1)
    requests['treq'] = (requests.pickup_datetime - requests.pickup_datetime.min())
    requests['ttrav'] = requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
    requests.tarr = [request.pickup_datetime + request.ttrav for _, request in requests.iterrows()]
    requests = requests.sort_values('treq')

    if logger is not None:
        logger.info('Input data prepared')

    return requests


def translate_to_osmnx(
        requests: pd.DataFrame,
        city_graph: nx.MultiDiGraph,
        **kwargs
) -> (pd.DataFrame, dict):
    """
    Amend an input dataset: convert dataframe with nodes'
    longitudes and latitudes to the osmnx ids.
    ---
    @param requests: dataframe with long, lat
    @param city_graph: graph of the city
    @param kwargs: names of the columns with longs and lats
    @return: .csv with origin and destinations in osmnx format
    ---
    Example:
    ---
    import pandas as pd
    import osmnx as ox
    import ExMAS

    params_chicago = ExMAS.utils.get_config(PATH)
    requests = pd.read_csv(params_chicago.paths.chicago_requests)

    df, loc_dict = ExMAS.utils.translate_to_osmnx(
        raw_data=requests,
        city_graph=ox.load_graphml(PATH),
        location_dictionary=None,
        origin_lat="Pickup Centroid Latitude",
        origin_long="Pickup Centroid Longitude",
        destination_long="Dropoff Centroid Longitude",
        destination_lat="Dropoff Centroid Latitude"
    )
    """
    origin_long = kwargs.get("origin_long", "ORIGIN_BLOCK_LONGITUDE")
    origin_lat = kwargs.get("origin_lat", "ORIGIN_BLOCK_LATITUDE")
    destination_long = kwargs.get("destination_long", "DESTINATION_BLOCK_LONG")
    destination_lat = kwargs.get("destination_lat", "DESTINATION_BLOCK_LAT")

    df = requests.copy()
    df = df.astype({col: float for col in [origin_long, origin_lat, destination_long, destination_lat]})
    city_graph = nx.relabel_nodes(city_graph, {node: int(node) for node in city_graph.nodes})

    df["origin"] = df.apply(
        lambda x: ox.distance.nearest_nodes(city_graph, x[origin_long], x[origin_lat]),
        axis=1
    )
    df["destination"] = df.apply(
        lambda x: ox.nearest_nodes(city_graph, x[destination_long], x[destination_lat]),
        axis=1
    )

    df = df.loc[df['origin'] != df['destination']]

    return df


def add_noise_to_data(
        requests: pd.DataFrame,
        skim: pd.DataFrame,
        distance: float or int = 300,
        **kwargs
) -> pd.DataFrame:
    """
    Sample nearby nodes for given centres of areas and sample
    time for the intervals. The feature created to amend datasets
    where only approximate locations and times are known
    ---
    @param requests: requests
    @param skim: dataframe with distances (indexes and columns
    are osmnx nodes)
    @param distance: distance from the centre points (in metres).
    The points will be substituted with nodes from this range
    @param kwargs: time specifics (hours, minutes, seconds):
    provide the length of the time interval from which
    you want to sample time
    @return: amended dataframe with the additional noise
    ---
    Example
    ---
    import pandas as pd
    import osmnx as ox
    import ExMAS

    params_chicago = ExMAS.utils.get_config(PATH)
    requests = pd.read_csv(params_chicago.paths.chicago_requests)
    skim = skim = pd.read_csv(PATH, index_col="Unnamed: 0")
    requests = ExMAS.utils.add_noise_to_data(requests, skim, minutes=15)
    """
    requests['initial_origin'] = requests['origin']
    requests['initial_destination'] = requests['destination']
    node_dict = {}

    skim.index = [int(t) for t in skim.index]
    skim.columns = [int(t) for t in skim.columns]

    node_list = list(requests['origin']) + list(requests['destination'])

    if "tqdm" in sys.modules:
        pbar = tqdm(total=len(node_list))

    for node in node_list:
        df_node = skim.loc[skim[node] < distance]

        if "tqdm" in sys.modules:
            pbar.update(1)

        if df_node.empty:
            node_dict[node] = [node]
        else:
            node_dict[node] = list(df_node.index)

    requests['origin'] = \
        requests['origin'].apply(lambda x: random.sample(node_dict[x], 1)[0])
    requests['destination'] = \
        requests['destination'].apply(lambda x: random.sample(node_dict[x], 1)[0])

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
            timedelta(minutes=np.random.randint(0, kwargs.get('minutes', 0) + 1)))
    requests['pickup_datetime'] = \
        requests['pickup_datetime'].apply(lambda x: x +
            timedelta(seconds=np.random.randint(0, kwargs.get('seconds', 0) + 1)))

    return requests.sort_values("pickup_datetime")


def load_skim_graph(
        params: dotmap.DotMap
):
    if os.path.isfile(params.paths.skim):
        try:
            skim = pd.read_csv(params.paths.skim, index_col='Unnamed: 0')
        except UnicodeDecodeError:
            skim = pd.read_parquet(params.paths.skim)
        city_graph = ox.load_graphml(params.paths.G)
    else:
        city_graph, skim = download_graph_skim(params.city)
        ox.save_graphml(city_graph, params.paths.G)
        skim.to_parquet(params.paths.skim)
    skim.columns = [int(c) for c in skim.columns]

    return city_graph, skim


def amend_washington_requests(
        raw_data,
        random_state: np.random._generator.Generator or None = None,
        **kwargs
):
    if random_state is not None:
        pass
    else:
        random_state = np.random.default_rng(secrets.randbits(kwargs.get('seed', 123)))
    df = raw_data.copy()
    df = df.loc[df["ORIGINCITY"].isin(["WASHINGTON", "Washington"])]
    df = df.loc[df["DESTINATIONCITY"].isin(["WASHINGTON", "Washington"])]

    for col_name in ["ORIGIN_BLOCK_LATITUDE", "ORIGIN_BLOCK_LONGITUDE",
                     "DESTINATION_BLOCK_LAT", "DESTINATION_BLOCK_LONG"]:
        df = df.loc[[not x for x in np.isnan(df[col_name])]]

    def temp_foo():
        x = random_state.integers(0, 59)
        return str(x) if x >= 10 else "0" + str(x)

    df["pickup_datetime"] = df.apply(lambda x: x["ORIGINDATETIME_TR"][:-2] +
                                               temp_foo(), axis=1)

    df = df.sort_values(by="pickup_datetime", ignore_index=True)

    return df


def load_washington_requests(
        params: dotmap.DotMap or dict,
        batch_size: int,
        start_time: pd.Timestamp = pd.Timestamp('2024-01-08 16-00'),
        interval_length_minutes: int = 30,
        random_state: np.random._generator.Generator or None = None,
        **kwargs
):
    """
    Read dataset downloaded directly from gov site of Washington DC,
    e.g. https://dcgov.app.box.com/v/TaxiTrips2024
    :param params: Washington config file
    :param batch_size: demand size
    :param start_time: starting time in pd.Timestamp
    :param interval_length_minutes: time interval for the demand
    :param random_state: for sampling purposes
    :param kwargs: other parameters if required
    :return:
    """
    requests = pd.read_csv(params['paths']['requests'])
    city_graph, skim = load_skim_graph(params)
    requests = amend_washington_requests(
        raw_data=requests, city_graph=city_graph, random_state=random_state)

    requests['pickup_datetime'] = pd.to_datetime(requests['pickup_datetime'])
    requests = requests.loc[requests['pickup_datetime'] >= start_time]
    requests = requests.loc[requests['pickup_datetime'] <= start_time +
                            pd.Timedelta(minutes=interval_length_minutes)]

    requests = translate_to_osmnx(
        requests=requests,
        city_graph=city_graph
    )
    requests = amend_demand_structure(
        requests=requests,
        skim_matrix=skim,
        params=params
    )

    if len(requests) > batch_size:
        requests = requests.sample(n=batch_size, random_state=random_state)
        requests['pax_id'] = range(len(requests))
        requests['id'] = requests['pax_id']
        requests.index = range(len(requests))
    else:
        raise ValueError(f"Requested dataset lacks sufficient demand:"
                         f"requested {batch_size}, demand size: {len(requests)}")

    return {'skim': skim,
            'requests': requests}

