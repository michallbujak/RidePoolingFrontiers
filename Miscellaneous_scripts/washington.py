import os
import pandas as pd
import dotmap
import logging
import ExMAS
import networkx as nx
import osmnx as ox
import numpy as np
import pickle
import datetime as dt
from tqdm import tqdm
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

    df = df.sort_values(by="pickup_datetime", ignore_index=True)

    return df


config = dotmap.DotMap()
config.path_results = 'Miscellaneous_scripts/data/'
utils.create_results_directory(config)

os.chdir(os.path.dirname(os.getcwd()))

params_dc = ExMAS.utils.get_config("Miscellaneous_scripts/configs/washington_init_config.json")
params_chicago = ExMAS.utils.get_config("Miscellaneous_scripts/configs/washington_init_config.json")


dotmaps_list_results = []

params_chicago.paths.requests = "ExMAS/data/chicago_2023_10k.csv"
requests = pd.read_csv(params_chicago.paths.requests)

np.random.seed(123)

requests["pickup_datetime"] = pd.to_datetime(requests["pickup_datetime"])

grouped_requests = requests.groupby(pd.Grouper(key='pickup_datetime', freq='30min'))
pbar = tqdm(total=len(grouped_requests.groups.keys()))

for key in grouped_requests.groups.keys():
    requests_batch = grouped_requests.get_group(key)

    requests_batch = requests_batch.sample(frac=0.3)

    try:
        dotmap_data, params_chicago = create_input_dotmap(requests_batch, params_chicago)
        results = ExMAS.main(dotmap_data, params_chicago, False)

        _results = results.copy()

        dotmaps_list_results.append((_results.sblts.requests,
                                     _results.sblts.schedule,
                                     _results.sblts.res,
                                     _results.sblts.rides))
    except:
        pass

    pbar.update(1)


final_results = [{"requests": x[0],
                  "schedule": x[1],
                  "results": x[2],
                  "rides": x[3]} for x in dotmaps_list_results]

with open("results_washington.pickle", "wb") as f:
    pickle.dump(final_results, f)

# for num in range(7, 13):
#     dotmaps_list_results = []
#     _ss = str(num) if num >= 10 else "0" + str(num)
#
#     params_dc.paths.requests = "ExMAS/data/washington_dc_requests_2019/taxi_2019_" + _ss + ".csv"
#     requests = pd.read_csv(params_dc.paths.requests)
#
#     np.random.seed(123)
#
#     requests["pickup_datetime"] = pd.to_datetime(requests["pickup_datetime"])
#
#     grouped_requests = requests.groupby(pd.Grouper(key='pickup_datetime', freq='30min'))
#     pbar = tqdm(total=len(grouped_requests.groups.keys()))
#
#     for key in grouped_requests.groups.keys():
#         requests_batch = grouped_requests.get_group(key)
#
#         requests_batch = requests_batch.sample(frac=0.3)
#
#         try:
#             dotmap_data, params_dc = create_input_dotmap(requests_batch, params_dc)
#             results = ExMAS.main(dotmap_data, params_dc, False)
#
#             _results = results.copy()
#
#             dotmaps_list_results.append((_results.sblts.requests,
#                                          _results.sblts.schedule,
#                                          _results.sblts.res,
#                                          _results.sblts.rides))
#         except:
#             pass
#
#         pbar.update(1)
#
#
#     final_results = [{"requests": x[0],
#                       "schedule": x[1],
#                       "results": x[2],
#                       "rides": x[3]} for x in dotmaps_list_results]
#
#     with open("results_washington_" + str(num) + ".pickle", "wb") as f:
#         pickle.dump(final_results, f)



# download
# exmas_params = DotMap()
# exmas_params.city = "Chicago, USA"
# exmas_params.dist_threshold = 10000
# exmas_params.paths.G = r"C:\Users\szmat\Documents\GitHub\RidePoolingFrontiers\ExMAS\data\graphs\Chicago.graphml"
# exmas_params.paths.skim = r"C:\Users\szmat\Documents\GitHub\RidePoolingFrontiers\ExMAS\data\graphs\Chicago.csv"
#
# dm = DotMap()
#
# def download_G(inData, _params, make_skims=True):
#     # uses osmnx to download the graph
#     inData.G = ox.graph_from_place(_params.city, network_type='drive')
#     inData.nodes = pd.DataFrame.from_dict(dict(inData.G.nodes(data=True)), orient='index')
#     if make_skims:
#         inData.skim_generator = nx.all_pairs_dijkstra_path_length(inData.G,
#                                                                   weight='length',
#                                                                   cutoff=_params.dist_threshold)
#         inData.skim_dict = dict(inData.skim_generator)  # filled dict is more usable
#         inData.skim = pd.DataFrame(inData.skim_dict).fillna(_params.dist_threshold).T.astype(
#             int)  # and dataframe is more intuitive
#     return inData
#
# def save_G(inData, _params, path=None):
#     # saves graph and skims to files
#     ox.save_graphml(inData.G, filepath=_params.paths.G)
#     inData.skim.to_csv(_params.paths.skim, chunksize=20000000)
#
# dm = download_G(dm, exmas_params)
# save_G(dm, exmas_params)