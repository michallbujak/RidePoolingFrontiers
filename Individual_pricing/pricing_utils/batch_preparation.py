import os
import sys
import json
import logging
import warnings

from dotmap import DotMap
import pandas as pd
from collections.abc import Callable
from tqdm import tqdm
import datetime

from ExMAS.utils import load_G

up = os.path.dirname
sys.path.append(up(up(up(__file__))))


def log_func(log, msg):
    if log is None:
        print(msg)
    else:
        log.info(msg)


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


def create_results_directory(config, date=None):
    if date is None:
        the_day = str(datetime.date.today().strftime("%d-%m-%y"))
    else:
        the_day = date
    config.path_results += the_day
    try:
        os.mkdir(config.path_results)
    except OSError as error:
        print(error)
        print('overwriting current files in the folder')
    try:
        os.mkdir(os.path.join(config.path_results, 'temp'))
    except OSError as error:
        print(error)
        print('temp folder already exists')
    config.path_results += '/'
    return config


def initialise_input_dotmap():
    databank_dotmap = DotMap()
    databank_dotmap['passengers'] = pd.DataFrame(columns=['id', 'pos', 'status'])
    databank_dotmap.passengers = databank_dotmap.passengers.set_index('id')
    databank_dotmap['requests'] = pd.DataFrame(
        columns=['pax', 'origin', 'destination', 'treq',
                 'tdep', 'ttrav', 'tarr', 'tdrop']
    ).set_index('pax')

    return databank_dotmap


def nyc_csv_prepare_batches(
        _params: DotMap
):
    try:
        _params.paths.nyc_requests
    except AttributeError as exc:
        raise Exception("no nyc trips data path specified") from exc

    trips = pd.read_csv(_params.paths.nyc_requests, index_col=0)  # load csv (prepared in the other notebook)
    trips.pickup_datetime = pd.to_datetime(trips.pickup_datetime)  # convert to times

    batches = trips.groupby(pd.Grouper(key='pickup_datetime', freq=_params.get('freq', '10min')))
    return batches, trips


def nyc_pick_batch(batches, trips, inData, _params, batch_no):
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
        _inData.requests = requests
        _inData.passengers.pos = _inData.requests.origin
        _params.nP = _inData.requests.shape[0]
    return _inData


def prepare_batches(
        general_config: DotMap,
        params: DotMap,
        filter_function: Callable[[int], bool] = lambda x: len(x.requests) > 0,
        output_params: bool = True,
        logger: logging.Logger or None = None
) -> list[DotMap]:
    """ Prepare batches from the NYC request file"""

    databank_dotmap = initialise_input_dotmap()
    try:
        databank_dotmap = load_G(databank_dotmap, params, stats=True)
    except FileNotFoundError:
        for _name in ["G", "skim", "nyc_requests"]:
            params.paths[_name] = os.path.join(up(up(up(__file__))), params.paths[_name])

        databank_dotmap = load_G(databank_dotmap, params, stats=True)

    batches, trips = nyc_csv_prepare_batches(params)

    log_func(logger, "Preparing NYC batches \n")

    out_data = []
    pbar = tqdm(total=general_config.no_replications)
    counter = 0
    batch_no = 0

    while counter < general_config.no_replications and batch_no < 8736:
        try:
            temp = nyc_pick_batch(batches, trips, databank_dotmap, params, batch_no)
            if filter_function(temp):
                out_data.append(temp)
                pbar.update(1)
                counter += 1
            else:
                log_func(logger, f'Batch no: {batch_no} skipped due to filter')

            batch_no += 1

        except AttributeError:
            log_func(logger, f'Impossible to attach batch number: {batch_no}')
            batch_no += 1

    pbar.close()

    log_func(logger, f"Batches READY! \n")

    if output_params:
        return out_data, params
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
