"""
Script with function designed to save/load files in main adaptive pricing script
split for new script for the code clarity in the main script
"""
import ast

import numpy as np
import json
import pandas as pd

from Dynamic_pricing.adaptive_main_clean import demand
from Individual_pricing.pricing_utils.batch_preparation import create_directory


def save_load_data(
        step: int,
        save_load: str='save',
        **kwargs
):
    assert save_load in ['save', 'load'], "save_load should be either 'save' or 'load'"

    if step == 0:
        assert kwargs['run_config'], 'run_config needs to be passed'
        assert kwargs['demand'], 'demand needs to be passed'
        assert kwargs['vot_sample'], 'vot_sample needs to be passed'
        assert kwargs['exmas_params'], 'exmas_params needs to be passed'
        assert kwargs['actual_class_membership'], 'actual_class_membership needs to be passed'

        if save_load == 'save':
            create_directory(kwargs['run_config'].path_results + 'Step_0')
            folder = kwargs['run_config'].path_results + 'Step_0/'
            all_requests = kwargs['demand']['exmas']['requests']
            all_rides = kwargs['demand']['exmas']['rides']

            all_requests.to_csv(folder + 'demand_sample_' + str(kwargs['run_config'].batch_size) + '.csv',
                                index=False)
            all_rides.to_csv(folder + 'rides' + '_' + str(kwargs['run_config'].batch_size) + '.csv', index=False)
            np.save(folder + 'sample' + '_' + str(kwargs['run_config'].sample_size), kwargs['vot_sample'])
            with open(folder + 'exmas_config.json', 'w') as _file:
                json.dump(kwargs['exmas_params'], _file)
            with open(folder + 'class_memberships.json', 'w') as _file:
                json.dump({k: str(v) for k, v in kwargs['actual_class_membership'].items()}, _file)

        if save_load == 'load':
            folder = kwargs['run_config'].path_results + 'Step_0/'
            all_requests = pd.read_csv(folder + 'demand_sample_' + str(kwargs['run_config'].batch_size) + '.csv')
            all_rides = pd.read_csv(folder + 'rides' + '_' + str(kwargs['run_config'].batch_size) + '.csv',
                                    converters={k: ast.literal_eval for k in
                                                ['indexes', 'u_paxes', 'individual_times', 'individual_distances']})
            vot_sample = np.load(folder + 'sample' + '_' + str(kwargs['run_config'].sample_size) + '.npy')
            with open(folder + 'exmas_config.json', 'r') as _file:
                exmas_params = json.load(_file)
            with open(folder + 'class_memberships.json', 'r') as _file:
                actual_class_membership = json.load(_file)
            actual_class_membership = {int(k): int(v) for k, v in actual_class_membership.items()}

            vot_sample_dict = {k: [t[0] for t in vot_sample if t[1] == k] for k in set(vot_sample[:, 1])}

            return ({'requests':all_requests, 'rides':all_rides}, vot_sample,
                    vot_sample_dict, exmas_params, actual_class_membership)

    return None

