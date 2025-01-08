""" Functions designed to load and stochastically adjust original
 TLC NYC data for ExMAS-based calculations """

import secrets

import numpy as np
import pandas as pd
import geopandas
import osmnx as ox
from tqdm import tqdm
from dotmap import DotMap

def initial_nyc_sampling_amendment(
        taxi_map_path: str,
        original_requests_path: str,
        **kwargs
) -> pd.DataFrame or None:
    """
    Load current trip data from the official NYC TLC database in parquet.
    Sample nodes to OSMNX mapping.
    ------
    :param taxi_map_path: path to mapping of zones to polygons in NYC
    :param original_requests_path: original request file form the TLC page
    :param kwargs: optional arguments such as to save the amended dataframe ('save'),
    provide a desired save path ('save_path'), return the amended dataframe ('return_dataframe'),
    specify sampling seed ('seed')
    :return dataframe with three main columns: origin, destination, request time
    """
    nyc_taxi_map = geopandas.read_file(taxi_map_path)
    manhattan_taxi_map = nyc_taxi_map.loc[nyc_taxi_map['borough'] == 'Manhattan']
    locations_manhattan = set(manhattan_taxi_map.index)

    requests = pd.read_parquet(original_requests_path)
    manhattan_requests = requests.loc[
        requests.apply(lambda req:
                       (req['PULocationID'] in locations_manhattan)&
                       (req['DOLocationID'] in locations_manhattan),
                       axis=1)
    ]

    # Now we proceed to sampling nodes within zones
    progress_bar = tqdm(total=len(manhattan_taxi_map))
    sampled_nodes = {}
    for _, zone in manhattan_taxi_map.iterrows():
        try:
            zone_graph = ox.graph_from_polygon(polygon=zone['geometry'], network_type='drive')
            sampled_nodes[_] = zone_graph.nodes
        except ox._errors.InsufficientResponseError:
            sampled_nodes[_] = []
        except ValueError:
            sampled_nodes[_] = []
        progress_bar.update(1)

    # remove zones with no nodes within
    empty_zones = {zone_key for zone_key, zone_nodes in sampled_nodes.items() if len(zone_nodes) == 0}
    manhattan_requests = manhattan_requests.loc[
        requests.apply(lambda req:
                       (req['PULocationID'] not in empty_zones) &
                       (req['DOLocationID'] not in empty_zones),
                       axis=1)
    ]

    # sample origins and destinations from the sampled nodes
    rng = np.random.default_rng(secrets.randbits(kwargs.get('seed', 123)))

    sampled_origins = [rng.choice(sampled_nodes[orig]) for orig in manhattan_requests['PULocationID']]
    sampled_destinations = [rng.choice(sampled_nodes[dest]) for dest in manhattan_requests['DOLocationID']]
    manhattan_requests['origin'] = sampled_origins
    manhattan_requests['destination'] = sampled_destinations
    manhattan_requests = manhattan_requests.rename(columns={'tpep_pickup_datetime': 'time_request'})
    manhattan_requests = manhattan_requests[['origin', 'destination', 'time_request']]

    # optionally save the file
    if kwargs.get('save', False):
        manhattan_requests.to_parquet(kwargs.get('save_path', 'manhattan_requests.parquet'))

    if kwargs.get('return_dataframe', True):
        return manhattan_requests
    else:
        return None


def adjust_nyc_request_to_exmas(
        nyc_requests_path: str,
        skim_matrix_path: str,
        batch_size: int,
        start_time: pd.Timestamp = pd.Timestamp('2024-01-08 16-00'),
        interval_length_minutes: int = 30,
        **kwargs
) -> dict or DotMap:
    """
    Load amended nyc requests in parquet. Subsample data
    to requested size and construct data in the format required
    by the ExMAS algorithm
    -------------------------
    :param nyc_requests_path: path to NYC trip requests after
    amendments with function initial_nyc_sampling_amendment
    :param skim_matrix_path: path to skim matrix
    :param batch_size: desired size of the demand for the whole
    interval length
    :param start_time: starting time of the batch
    :param interval_length_minutes: in minutes the interval time on which
    the requests should be collected
    :param kwargs: other parameters if required (e.g. seed)
    :return:
    """
    # Load data
    requests = pd.read_parquet(nyc_requests_path)
    try:
        skim_matrix = pd.read_csv(skim_matrix_path)
    except UnicodeDecodeError:
        skim_matrix = pd.read_parquet(skim_matrix_path)

    # Filter
    end_time = start_time + pd.Timedelta(minutes=interval_length_minutes)
    requests = requests.loc[
        [(t >= start_time)&(t<end_time) for t in requests['time_request']]
    ]
    if len(requests) < batch_size:
        raise ValueError('The desired batch size is greater than number '
                         'of requests in the given interval.'
                         'Either change interval or lower "batch_size".')

    # Sample
    rng = np.random.default_rng(secrets.randbits(kwargs.get('seed', 123)))


    requests['dist'] = requests.apply(
        lambda request: skim_matrix.loc[request.origin, request.destination],
        axis=1
    )
    requests['treq'] = requests['time_request'] - min(requests['time_request'])
    requests['ttrav'] = requests.apply(
        lambda request: pd.Timedelta(request.dist, 's').floor('s'),
        axis=1
    )
    requests['pax_id'] = requests.index.copy()

    return {}
