""" Run ExMAS in probabilistic settings"""

import os
import pandas as pd
import numpy as np

import Utils.utils_topology as utils
from ExMAS.probabilistic_exmas import main as exmas_algo
import utils.batch_preparation as bt_prep
from utils.exmas_loop import exmas_loop_func

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

databanks_list, settings_list = exmas_loop_func(
    exmas_algorithm=exmas_algo,
    params=params,
    list_databanks=databanks_list,
    topo_params=general_config,
    replications=general_config.no_replications,
    logger=None,
    sampling_function_with_index=True
)