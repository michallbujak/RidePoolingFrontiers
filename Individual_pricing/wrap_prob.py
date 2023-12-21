import pickle

from ExMAS.probabilistic_exmas import main as exmas_algo
import Individual_pricing.pricing_utils.batch_preparation as bt_prep
from Individual_pricing.exmas_loop import exmas_loop_func
from Individual_pricing.matching import matching_function
from Individual_pricing.evaluation import *
from Individual_pricing.pricing_functions import *

# general_config = bt_prep.get_parameters(
#     "configs/general_config.json"
# )
#
# bt_prep.create_results_directory(general_config)
#
# databanks_list, params = bt_prep.prepare_batches(
#     general_config=general_config,
#     params=bt_prep.get_parameters(general_config.initial_parameters),
#     filter_function=lambda x: (len(x.requests) < 150) &
#                               (len(x.requests) > 140)
# )
#
# params = bt_prep.update_probabilistic(general_config, params)
# params.type_of_distribution = None
#
# databanks_list, settings_list = exmas_loop_func(
#     exmas_algorithm=exmas_algo,
#     params=params,
#     list_databanks=databanks_list,
#     topo_params=general_config,
#     replications=general_config.no_replications,
#     logger=None,
#     sampling_function_with_index=False
# )
#
# with open("example_data", "wb") as file:
#     pickle.dump((databanks_list, settings_list, params), file)

with open("example_data", "rb") as file:
    databanks_list, settings_list, params = pickle.load(file)

databanks_list = [expand_rides(t) for t in databanks_list]

databanks_list = [prepare_samples(t, 20) for t in databanks_list]

databanks_list = [
    calculate_expected_profitability(
        t,
        final_sample_size=10,
        price=params["price"]/1000
    )
    for t in databanks_list
]

x = 0

