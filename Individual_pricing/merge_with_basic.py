from ExMAS.probabilistic_exmas import main as exmas_algo
import Individual_pricing.pricing_utils.batch_preparation as bt_prep
from Individual_pricing.exmas_loop import exmas_loop_func
from Individual_pricing.matching import matching_function
from Individual_pricing.evaluation import *
from Individual_pricing.pricing_functions import *

directories = bt_prep.get_parameters("Individual_pricing/configs/directories.json")
bt_prep.create_results_directory(directories, "test_v2")


databanks_list, exmas_params = bt_prep.prepare_batches(
    exmas_params=bt_prep.get_parameters(directories.initial_parameters),
    filter_function=lambda x: len(x.requests) == 150
)
for discount in [0.12, 0.2]:
    exmas_params["shared_discount"] = discount
    exmas_params.type_of_distribution = None
    databanks_list = exmas_loop_func(
        exmas_algorithm=exmas_algo,
        exmas_params=exmas_params,
        list_databanks=databanks_list
    )

