import os
import pandas as pd
import dotmap
import logging
import ExMAS
import networkx as nx
import osmnx as ox
import numpy as np
import pickle
from Utils import utils_topology as utils


def create_input_dotmap(
        requests: pd.DataFrame,
        params: dotmap.DotMap,
        logger: logging.Logger or None = None,
        **kwargs
) -> (dotmap.DotMap, dotmap.DotMap):
    """
    Function designed to translate a dataframe with origins, destinations and times
    to the suitable ExMAS format
    @param requests: dataframe with columns including origin, destination and
    request time. If non-standard column names, refer to kwargs
    @param params: configuration
    @param logger: if you want to log the activities, pass logger
    @param kwargs: special names for columns: origin_name, destination_name,
     pickup_datetime_name
    @return: inputs for the ExMAS main()
    ---
    Example:
    params_nyc = ExMAS.utils.get_config(PATH)

    requests_full = pd.read_csv(params_nyc.paths.nyc_requests)
    requests_short = requests_full.loc[requests_full.index < 100]

    dotmap_data, params_nyc = create_input_dotmap(requests_short, params_nyc)
    results = ExMAS.main(dotmap_data, params_nyc, False)
    """

    dataset_dotmap = dotmap.DotMap()
    dataset_dotmap['passengers'] = pd.DataFrame(columns=['id', 'pos', 'status'])
    dataset_dotmap.passengers = dataset_dotmap.passengers.set_index('id')
    dataset_dotmap['requests'] = pd.DataFrame(
        columns=['pax', 'origin', 'destination', 'treq', 'tdep', 'ttrav', 'tarr', 'tdrop']) \
        .set_index('pax')
    dataset_dotmap = ExMAS.utils.load_G(dataset_dotmap, params, stats=True)

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

    # df["origin"] = df.apply(lambda x:
    #                         ox.nearest_nodes(
    #                             city_graph,
    #                             float(x[origin_long]),
    #                             float(x[origin_lat])
    #                         ),
    #                         axis=1)
    # df["destination"] = df.apply(lambda x:
    #                              ox.nearest_nodes(
    #                                  city_graph,
    #                                  x[destination_long],
    #                                  x[destination_lat]
    #                              ),
    #                              axis=1)

    return df, location_dictionary


def amend_washington_requests(raw_data, seed=123):
    np.random.seed(seed)
    df = raw_data.copy()
    df = df.loc[df["ORIGINCITY"].isin(["WASHINGTON", "Washington"])]
    df = df.loc[df["DESTINATIONCITY"].isin(["WASHINGTON", "Washington"])]

    for col_name in ["ORIGIN_BLOCK_LATITUDE", "ORIGIN_BLOCK_LONGITUDE",
                     "DESTINATION_BLOCK_LAT", "DESTINATION_BLOCK_LONG"]:
        df = df.loc[[not x for x in np.isnan(df[col_name])]]

    def temp_foo():
        x = np.random.randint(0, 59)
        return str(x) if x >= 10 else "0" + str(x)

    df["pickup_datetime"] = df.apply(lambda x: x["ORIGINDATETIME_TR"][:-2] +
                                               temp_foo(), axis=1)

    return df


config = dotmap.DotMap()
config.path_results = 'Miscellaneous_scripts/data/'
utils.create_results_directory(config)

os.chdir(os.path.dirname(os.getcwd()))

params_dc = ExMAS.utils.get_config("Miscellaneous_scripts/configs/washington_init_config.json")

with open("ExMAS/data/washington_dc_requests/washington_osmnx_ids.pickle", "rb") as f:
    location_dict = pickle.load(f)

for num in range(2, 7):
    params_dc.paths.requests = "ExMAS/data/OpenDataDC_Taxi_2022/taxi_20220" + str(num) + ".csv"

    requests_full = pd.read_csv(params_dc.paths.requests)
    requests_amended = amend_washington_requests(requests_full)
    requests_amended, location_dict = translate_to_osmnx(
        raw_data=requests_amended,
        city_graph=ox.load_graphml(params_dc.paths.G),
        location_dictionary=location_dict
    )

    requests_amended.to_csv('ExMAS/data/washington_dc_requests/taxi_20220' + str(num) + '.csv')

    with open("ExMAS/data/washington_dc_requests/washington_osmnx_ids.pickle", "wb") as f:
        pickle.dump(location_dict, f)


# dotmap_data, params_nyc = create_input_dotmap(requests_amended.loc[:100], params_dc)
# results = ExMAS.main(dotmap_data, params_nyc, False)
