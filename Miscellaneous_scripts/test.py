import pandas as pd
import osmnx as ox
import networkx as nx
from dotmap import DotMap
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pickle
import dotmap
from Utils import utils_topology as utils
import os
import ExMAS
from tqdm import tqdm
import logging
import time


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
    @param requests: dataframe with columns including origin, destination and
    request time. If non-standard column names, refer to kwargs
    @param params: configuration
    @param data_provided: provide data for mulitple comp
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

    if data_provided is None:
        dataset_dotmap = ExMAS.utils.load_G(dataset_dotmap, params, stats=True)
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


config = dotmap.DotMap()
config.path_results = 'Miscellaneous_scripts/data/'
utils.create_results_directory(config)

os.chdir(os.path.dirname(os.getcwd()))

params_chicago = ExMAS.utils.get_config("Miscellaneous_scripts/configs/chiFcago_init_config.json")

dotmaps_list_results = []

params_chicago.paths.requests = "ExMAS/data/chicago_2023_amended_feb.csv"
_requests = pd.read_csv(params_chicago.paths.requests)

np.random.seed(123)

_requests["pickup_datetime"] = pd.to_datetime(_requests["pickup_datetime"])

grouped_requests = _requests.groupby(pd.Grouper(key='pickup_datetime', freq='30min'))

graph = ox.load_graphml(filepath=params_chicago.paths.G)
nodes = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')
skim = pd.read_csv(params_chicago.paths.skim, index_col='Unnamed: 0')
skim.columns = [int(c) for c in skim.columns]
graphs_data = {"G": graph, "skim": skim, "nodes": nodes}

pbar = tqdm(total=len(grouped_requests.groups.keys()))

for num, key in enumerate(grouped_requests.groups.keys()):
    time1 = time.time()
    requests_batch = grouped_requests.get_group(key)

    requests_batch = requests_batch.sample(frac=0.3)

    if num == 1030:
        break

    try:
        dotmap_data, params_chicago = create_input_dotmap(
            requests=requests_batch,
            params=params_chicago,
            data_provided=graphs_data
        )

        time2 = time.time()
        print(f"Request size: {len(requests_batch)}, batch preparation time: {time2 - time1}")

        results = ExMAS.main(dotmap_data, params_chicago, False)

        time3 = time.time()
        print(f"Calculation time: {time3 - time2}")

        _results = results.copy()

        dotmaps_list_results.append((_results.sblts.requests,
                                     _results.sblts.schedule,
                                     _results.sblts.res,
                                     _results.sblts.rides))

        print("finished")
    except:
        pass

    pbar.update(1)

final_results = [{"requests": x[0],
                  "schedule": x[1],
                  "results": x[2],
                  "rides": x[3]} for x in dotmaps_list_results]

with open("results_chicago.pickle", "wb") as f:
    pickle.dump(final_results, f)
