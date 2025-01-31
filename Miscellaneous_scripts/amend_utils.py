import sys

import pandas as pd
import logging
import networkx as nx
import dotmap
import ExMAS
import osmnx as ox


def create_input_dotmap(
        requests: pd.DataFrame,
        params: dotmap.DotMap,
        data_provided: dict or None = None,
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
    @param data_provided: provide data for multiple computations,
    i.e. if you experiment many runs in the same city, it's efficient not
    to load each time city graph and skim matrix, but rather store it.
    The dictionary should contain: "G", "nodes" and "skim"
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

    dataset_dotmap = dotmap.DotMap()
    dataset_dotmap['passengers'] = pd.DataFrame(columns=['id', 'pos', 'status'])
    dataset_dotmap.passengers = dataset_dotmap.passengers.set_index('id')
    dataset_dotmap['requests'] = pd.DataFrame(
        columns=['pax', 'origin', 'destination', 'treq', 'tdep', 'ttrav', 'tarr', 'tdrop']) \
        .set_index('pax')

    dataset_dotmap = load_G(dataset_dotmap, params, stats=True)

    if data_provided is None:
        dataset_dotmap.G = ox.load_graphml(filepath=params.paths.G)
        dataset_dotmap.nodes = pd.DataFrame.from_dict(dict(dataset_dotmap.G.nodes(data=True)), orient='index')
        skim = pd.read_csv(params.paths.skim, index_col='Unnamed: 0')
        skim.columns = [int(c) for c in skim.columns]
        dataset_dotmap.skim = skim
    else:
        dataset_dotmap.G = data_provided["G"]
        dataset_dotmap.nodes = data_provided["nodes"]
        dataset_dotmap.skim = data_provided["skim"]

    pickup_time_name = kwargs.get('pickup_time_name', 'pickup_datetime')
    requests[pickup_time_name] = pd.to_datetime(requests[pickup_time_name])

    if 'origin' in requests.columns and 'destination' in requests.columns:
        pass
    else:
        requests['origin'] = kwargs.get('origin_name', 'origin')
        requests['destination'] = kwargs.get('destination_name', 'destination')

    requests['status'] = 0
    requests['requests'] = requests['origin']

    dataset_dotmap.passengers = requests.copy()

    requests['dist'] = requests.apply(lambda request:
                                      dataset_dotmap.skim.loc[request.origin, request.destination],
                                      axis=1)
    requests['treq'] = (requests.pickup_datetime - requests.pickup_datetime.min())
    requests['ttrav'] = requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
    requests.tarr = [request.pickup_datetime + request.ttrav for _, request in requests.iterrows()]
    requests = requests.sort_values('treq')
    requests['pax_id'] = requests.index.copy()

    dataset_dotmap.requests = requests
    dataset_dotmap.passengers.pos = dataset_dotmap.requests.origin
    params.nP = dataset_dotmap.requests.shape[0]

    if logger is not None:
        logger.info('Input data prepared')

    return dataset_dotmap, params


def translate_to_osmnx(
        raw_data: pd.DataFrame,
        city_graph: nx.MultiDiGraph,
        location_dictionary: dict or None = None,
        **kwargs
) -> (pd.DataFrame, dict):
    """
    Amend an input dataset: convert dataframe with nodes'
    longitudes and latitudes to the osmnx ids.
    ---
    @param raw_data: dataframe with long, lat
    @param city_graph: graph of the city
    @param location_dictionary: dictionary with locations
    the optional parameter, useful if you want to translate
    many request files corresponding to the same city
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
        requests['origin'].apply(lambda x: random.vot_sample(node_dict[x], 1)[0])
    requests['destination'] = \
        requests['destination'].apply(lambda x: random.vot_sample(node_dict[x], 1)[0])

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
