import os
import pandas as pd
import dotmap
import logging
import ExMAS
from Utils import utils_topology as utils

os.chdir(os.path.dirname(os.getcwd()))


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
        columns=['pax', 'origin', 'destination', 'treq', 'tdep', 'ttrav', 'tarr', 'tdrop'])\
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


config = utils.get_parameters('Miscellaneous_scripts/configs/test_config.json')
config.path_results = 'Miscellaneous_scripts/' + config.path_results
utils.create_results_directory(config)

params_nyc = ExMAS.utils.get_config(config.initial_parameters)

requests_full = pd.read_csv(params_nyc.paths.nyc_requests)
requests_short = requests_full.loc[requests_full.index < 100]

dotmap_data, params_nyc = create_input_dotmap(requests_short, params_nyc)
results = ExMAS.main(dotmap_data, params_nyc, False)
