""" Run ExMAS in probabilistic/deterministic settings"""
import pickle

from ExMAS.probabilistic_exmas import main as exmas_algo
import Individual_pricing.pricing_utils.batch_preparation as bt_prep
from Individual_pricing.exmas_loop import exmas_loop_func
from Individual_pricing.matching import matching_function
from Individual_pricing.evaluation import evaluate_pooling, \
    compare_objective_methods, aggregate_results

from Individual_pricing.pricing_functions import *

general_config = bt_prep.get_parameters(
    r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Individual_pricing\configs\general_config.json"
)

bt_prep.create_results_directory(general_config)

databanks_list, params = bt_prep.prepare_batches(
    general_config=general_config,
    params=bt_prep.get_parameters(general_config.initial_parameters),
    filter_function=lambda x: (len(x.requests) < 150) &
                              (len(x.requests) > 140)
)

params = bt_prep.update_probabilistic(general_config, params)
params.multinormal_probs = (0.29, 0.57, 0.81, 1)
params.multinormal_args = (
    ((16.98 / 3600, 1.22), (1.68 / 3600, 0.122)),
    ((14.02 / 3600, 1.135), (1.402 / 3600, 0.1135)),
    ((100 / 3600, 5), (2.625 / 3600, 0.105)),
    ((7.78 / 3600, 1.18), (0.778 / 3600, 0.118))
)

databanks_list, settings_list = exmas_loop_func(
    exmas_algorithm=exmas_algo,
    exmas_params=params,
    list_databanks=databanks_list,
    general_configuration=general_config,
    replications=general_config.no_replications,
    logger=None,
    sampling_function_with_index=True
)

params.sampling_function = None

with open("example_data_known", "wb") as file:
    pickle.dump((databanks_list, settings_list, params), file)

with open("example_data_known", "rb") as file:
    databanks_list, settings_list, params = pickle.load(file)

travellers_traits = [
    extract_travellers_data(
        databank=db,
        params=params
    ) for db in databanks_list
]

databanks_list = [expand_rides(t) for t in databanks_list]

databanks_list = [
    calculate_min_discount(
        databank=db,
        travellers_characteristics=tt
    ) for db, tt in zip(databanks_list, travellers_traits)
]

databanks_list = [
    calculate_profitability(
        databank=db,
        params=params
    ) for db in databanks_list
]

databanks_list = [
    matching_function(
        databank=db,
        params=params,
        objectives=[
            "profit_base",
            "profit_max",
            "profitability_base",
            "profitability_max"
        ]
    ) for db in databanks_list
]

databanks_list = [
    evaluate_pooling(db) for db in databanks_list
]

results = [
    compare_objective_methods(db)
    for db in databanks_list
]

z = 0

aggregated_results = aggregate_results(0)
