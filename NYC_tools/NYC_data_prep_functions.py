import pandas as pd
import random
from dotmap import DotMap
import json
import os, sys
from tqdm import tqdm
import ExMAS.utils
import logging


def initialise_indata_dotmap():
    inData = DotMap()
    inData['passengers'] = pd.DataFrame(columns=['id', 'pos', 'status'])
    inData.passengers = inData.passengers.set_index('id')
    inData['requests'] = pd.DataFrame(
        columns=['pax', 'origin', 'destination', 'treq', 'tdep', 'ttrav', 'tarr', 'tdrop']).set_index(
        'pax')
    return inData


def get_config(path, root_path=None):
    with open(path) as json_file:
        data = json.load(json_file)
        config = DotMap(data)
    config['t0'] = pd.Timestamp('15:00')

    if root_path is not None:
        config.paths.G = os.path.join(root_path, config.paths.G)  # graphml of a current .city
        config.paths.skim = os.path.join(root_path, config.paths.skim)  # csv with a skim between the nodes of the .city

    return config


def load_nyc_csv(_inData, _params):
    # loads the csv with trip requests
    # filter for the trips within a predefined time window (either exact, or a batch with a given frequency)
    try:
        _params.paths.nyc_requests
    except:
        raise Exception("no nyc trips data path specified")

    trips = pd.read_csv(_params.paths.nyc_requests, index_col=0)  # load csv (prepared in the other notebook)
    trips.pickup_datetime = pd.to_datetime(trips.pickup_datetime)  # convert to times

    # A: Filter for simulation times
    if _params.get('freq', 'False'):  # given frequency (default '10min')
        batches = trips.groupby(pd.Grouper(key='pickup_datetime', freq=_params.get('freq', '10min')))
        if _params.get('batch', 'False'):  # random batch
            batch = list(batches.groups.keys())[_params.batch]  # particular batch
        else:  # random 'freq'-minute batch# i-th batch
            batch = random.choice(list(batches.groups.keys()))
        df = batches.get_group(batch)
    else:  # exact date and sim-time
        early = pd.to_datetime(_params.date) + pd.to_timedelta(_params.t0 + ":00")
        late = pd.to_datetime(_params.date) + pd.to_timedelta(_params.t0 + ":00") + pd.to_timedelta(_params.simTime,
                                                                                                    unit='H')
        df = trips[(trips.pickup_datetime >= early) & (trips.pickup_datetime < late)]

    # B: Populate missing fields

    df['status'] = 0
    df.pos = df['origin']
    _inData.passengers = df
    requests = df
    requests['dist'] = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)
    requests['treq'] = (trips.pickup_datetime - trips.pickup_datetime.min())
    requests['ttrav'] = requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
    # requests.ttrav = pd.to_timedelta(requests.ttrav)
    # if params.get('avg_speed',False):
    #    requests.ttrav = (pd.to_timedelta(requests.ttrav) / _params.avg_speed).dt.floor('1s')
    requests.tarr = [request.pickup_datetime + request.ttrav for _, request in requests.iterrows()]
    requests = requests.sort_values('treq')
    requests['pax_id'] = requests.index.copy()
    _inData.requests = requests
    _inData.passengers.pos = _inData.requests.origin
    _params.nP = _inData.requests.shape[0]
    return _inData


def nyc_csv_prepare_batches(_inData, _params):
    try:
        _params.paths.nyc_requests
    except:
        raise Exception("no nyc trips data path specified")

    trips = pd.read_csv(_params.paths.nyc_requests, index_col=0)  # load csv (prepared in the other notebook)
    trips.pickup_datetime = pd.to_datetime(trips.pickup_datetime)  # convert to times

    batches = trips.groupby(pd.Grouper(key='pickup_datetime', freq=_params.get('freq', '10min')))
    return batches, trips


def nyc_pick_batch(batches, trips, inData, _params, batch_no):
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


def prepare_batches(number_of_batches, config_name="nyc_prob", filter_function=lambda x: len(x.requests) > 0,
                    freq="10min", output_params=True, logger=None):
    logger = embed_logger(logger)
    copy_wd = os.getcwd()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    params = get_config('ExMAS/data/configs/' + config_name + '.json')
    params.freq = freq
    inData = initialise_indata_dotmap()
    inData = ExMAS.utils.load_G(inData, params, stats=True)

    batches, trips = nyc_csv_prepare_batches(inData, params)

    logger.warning("Preparing NYC batches \n")
    inDatas = []
    for j in tqdm(range(number_of_batches)):
        try:
            temp = nyc_pick_batch(batches, trips, inData, params, j)
            if filter_function(temp):
                inDatas.append(temp)
            else:
                logger.debug('Impossible to attach batch number: ', j)
                pass
        except:
            pass

    os.chdir(copy_wd)
    logger.warning("Batches READY! \n")
    if output_params:
        return inDatas, params
    else:
        return inDatas


def run_exmas_nyc_batches(exmas_algorithm, params, indatas, replications=1, logger=None):
    logger = embed_logger(logger)
    results = []
    params.logger_level = "CRITICAL"
    logger.warning("Calculating ExMAS values \n ")
    for i in tqdm(range(len(indatas))):
        try:
            for j in range(replications):
                temp = exmas_algorithm(indatas[i], params, False)
                results.append(temp.copy())
        except:
            logger.debug('Impossible to attach batch number: ', i)
            pass

    logger.warning("Number of calculated results for batches is: ", len(results))
    logger.warning("ExMAS calculated \n")
    return results


def init_log(logger_level, logger=None):
    if logger_level == 'DEBUG':
        level = logging.DEBUG
    elif logger_level == 'WARNING':
        level = logging.WARNING
    elif logger_level == 'CRITICAL':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    if logger is None:
        logging.basicConfig(stream=sys.stdout, format='%(asctime)s-%(levelname)s-%(message)s',
                            datefmt='%H:%M:%S', level=level)

        logger = logging.getLogger()

        logger.setLevel(level)
        return logging.getLogger(__name__)
    else:
        logger.setLevel(level)
        return logger


def embed_logger(log):
    if log is None:
        return init_log('WARNING')
    elif not log:
        return init_log("CRITICAL")
    elif log:
        return init_log("DEBUG")
    else:
        return log
