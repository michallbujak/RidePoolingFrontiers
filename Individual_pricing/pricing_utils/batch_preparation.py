import os
import sys
import json
import logging
import warnings
import logging

from dotmap import DotMap
import pandas as pd
from collections.abc import Callable
from tqdm import tqdm
import datetime

from ExMAS.utils import load_G, download_G

up = os.path.dirname
sys.path.append(up(up(up(__file__))))


def log_func(
        log_level: int or str,
        msg: str,
        logger: logging.Logger or None = None
):
    """ Log things """
    if logger is None:
        logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s',
                            datefmt='%H:%M:%S')
        logger = logging.getLogger()
        logger.log(log_level, msg)
    else:
        logger.log(log_level, msg)


def get_parameters(
        path: str,
        time_correction: bool = False
) -> DotMap:
    with open(path) as json_file:
        data = json.load(json_file)
        config = DotMap(data)
    if time_correction:
        config['t0'] = pd.Timestamp('15:00')

    return config


def create_directory(
        directory: str,
        logger: logging.Logger or None = None,
        log_level: int or str = 20
):
    """ Create a directory with optional logging """
    try:
        os.mkdir(directory)
    except OSError:
        log_func(log_level,
                 f'Folder "{directory}" already exists and is set'
                 f' as results destination (warning: possible overwriting).',
                 logger)
    else:
        log_func(log_level,
                 f'Folder "{directory}" created and set for results.',
                 logger)


def create_results_directory(
        config: DotMap or dict,
        name: str or None = None,
        **kwargs
):
    if name is None:
        name = str(datetime.date.today().strftime("%d-%m-%y"))

    create_directory(config.path_results, kwargs.get('logger'), 10)
    config.path_results += name
    create_directory(config.path_results, kwargs.get('logger'), 20)
    config.path_results += '/'

    # create_directory(config.partial_results, kwargs.get('logger'), 10)
    # config.partial_results += name
    # create_directory(config.partial_results, kwargs.get('logger'), 20)
    # config.partial_results += '/'

    if kwargs.get("new_directories"):
        keys = ['initial_parameters', 'path_results', 'partial_results']
        dirs = {k: v for k, v in config.items() if k in keys}
        with open(kwargs["directories_path"][:-5] + "_" + name + ".json", "w") as f:
            json.dump(dirs, f)

    return config


def initialise_input_dotmap():
    """ Function required for the original ExMAS """
    databank_dotmap = DotMap()
    databank_dotmap['passengers'] = pd.DataFrame(columns=['id', 'pos', 'status'])
    databank_dotmap.passengers = databank_dotmap.passengers.set_index('id')
    databank_dotmap['requests'] = pd.DataFrame(
        columns=['pax', 'origin', 'destination', 'treq',
                 'tdep', 'ttrav', 'tarr', 'tdrop']
    ).set_index('pax')

    return databank_dotmap


def nyc_csv_prepare_batches(
        _params: DotMap,
        skim_nodes: list or None = None
):
    try:
        _params.paths.nyc_requests
    except AttributeError as exc:
        raise Exception("no nyc trips data path specified") from exc

    trips = pd.read_csv(_params.paths.nyc_requests, index_col=0)  # load csv (prepared in the other notebook)
    if skim_nodes:
        trips = trips.loc[[(t in skim_nodes)&(z in skim_nodes) for t, z in trips[['origin', 'destination']].values.tolist()]]
        trips.to_csv(_params.paths.nyc_requests)
    trips.pickup_datetime = pd.to_datetime(trips.pickup_datetime)  # convert to times

    batches = trips.groupby(pd.Grouper(key='pickup_datetime', freq=_params.get('freq', '10min')))
    return batches, trips


def nyc_pick_batch(batches, trips, inData, _params, batch_no, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _inData = inData.copy()
        batch = list(batches.groups.keys())[batch_no]
        df = batches.get_group(batch)
        df['status'] = 0
        df.pos = df['origin']
        _inData.passengers = df
        requests = df
        requests['dist'] = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)
        requests['treq'] = (trips.pickup_datetime - trips.pickup_datetime.min())
        requests['ttrav'] = requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
        requests.tarr = [request.pickup_datetime + request.ttrav for _, request in requests.iterrows()]
        requests = requests.sort_values('treq')
        requests['pax_id'] = requests.index.copy()
        requests = requests.loc[requests['dist'] > 0]
        _inData.requests = requests
        _inData.passengers.pos = _inData.requests.origin
        _params.nP = _inData.requests.shape[0]
    return _inData


def base_amend_requests(
        dotmap_data: DotMap or dict
):
    requests = dotmap_data['requests']
    requests['dist'] = requests.apply(lambda request:
                                      dotmap_data['skim'].loc[request.origin, request.destination],
                                      axis=1)
    requests['treq'] = (requests.pickup_datetime.apply(pd.to_datetime)
                        - requests.pickup_datetime.apply(pd.to_datetime).min())
    requests['pickup_datetime'] = requests['pickup_datetime'].apply(pd.to_datetime)
    requests['ttrav'] = requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
    requests['tarr'] = [request.pickup_datetime + request.ttrav for _, request in requests.iterrows()]
    requests = requests.sort_values('treq')
    requests['pax_id'] = requests.index.copy()
    requests = requests.loc[requests['dist'] > 0]
    return dotmap_data


def prepare_batches(
        exmas_params: DotMap,
        no_replications: int = 1,
        filter_function: Callable[[int], bool] = lambda x: len(x.requests) > 0,
        output_params: bool = True,
        **kwargs
) -> list[DotMap] or (list[DotMap], DotMap or dict):
    """ Prepare batches from the NYC request file"""
    databank_dotmap = initialise_input_dotmap()

    if no_replications == 1 & kwargs.get('quick_load'):
        req_name = 'nyc_demand_' + str(kwargs['batch_size']) + '.csv'
        try:
            databank_dotmap['requests'] = pd.read_csv(
                exmas_params.paths['nyc_requests'][:-16] + req_name)
            databank_dotmap = load_G(databank_dotmap, exmas_params)
            databank_dotmap = base_amend_requests(databank_dotmap)
        except FileNotFoundError:
            raise Warning('Quick load unsuccessful, check paths')
        if output_params:
            return databank_dotmap, exmas_params
        else:
            return databank_dotmap


    try:
        databank_dotmap = load_G(databank_dotmap, exmas_params, stats=True)
    except FileNotFoundError:
        log_func(30, "'G', 'skim' and 'nyc_requests' not found. "
                     "Checking parent directory",
                 kwargs.get('logger'))
        for _name in ["G", "skim", "nyc_requests"]:
            exmas_params.paths[_name] = os.path.join(up(up(up(__file__))),
                                                     exmas_params.paths[_name])

        try:
            databank_dotmap = load_G(databank_dotmap, exmas_params, stats=True)
        except FileNotFoundError:
            databank_dotmap = download_G(databank_dotmap, exmas_params)



    batches, trips = nyc_csv_prepare_batches(exmas_params) #

    logger = kwargs.get("logger")

    log_func(20, "Preparing NYC batches", logger)

    out_data = []
    pbar = tqdm(total=no_replications)
    counter = 0
    batch_no = 0

    while counter < no_replications and batch_no < 8736:
        try:
            temp = nyc_pick_batch(batches, trips, databank_dotmap, exmas_params, batch_no)
            if filter_function(temp):
                out_temp = {k: temp[k] for k in ["requests", "skim"]}
                out_data.append(out_temp)
                pbar.update(1)
                counter += 1
            else:
                log_func(10, f'Batch no: {batch_no} skipped due to filter', logger)

            batch_no += 1

        except AttributeError:
            log_func(10, f'Impossible to attach batch number: {batch_no}', logger)
            batch_no += 1

    pbar.close()

    log_func(20, f"Batches READY!", logger)

    if output_params:
        return out_data, exmas_params
    else:
        return out_data


def update_probabilistic(config, params):
    for k in ["distribution_variables",
              "type_of_distribution",
              "distribution_details",
              "panel_noise",
              "noise"]:
        params[k] = config.get(k, None)
    return params
