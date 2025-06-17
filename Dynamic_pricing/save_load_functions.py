"""
Script with function designed to save/load files in main adaptive pricing script
split for new script for the code clarity in the main script
"""
import ast
import pickle

import numpy as np
import json
import pandas as pd

from Individual_pricing.pricing_utils.batch_preparation import create_directory


def save_load_data(
        step: int,
        save_load: str='save',
        **kwargs
):
    assert save_load in ['save', 'load'], "save_load should be either 'save' or 'load'"

    if step == 0:
        assert kwargs['run_config'], 'run_config needs to be passed'

        if save_load == 'save':
            assert all(t in kwargs for t in
                       ['demand', 'vot_sample', 'exmas_params', 'actual_class_membership']),\
                'demand, vot_sample, exmas_params, actual_class_membership need to be passed'

            create_directory(kwargs['run_config'].path_results + 'Step_0')
            folder = kwargs['run_config'].path_results + 'Step_0/'

            kwargs['demand']['requests'].to_csv(folder + 'demand_sample_' + str(kwargs['run_config'].batch_size) + '.csv',
                                index=False)
            kwargs['demand']['rides'].to_csv(folder + 'rides' + '_' + str(kwargs['run_config'].batch_size) + '.csv', index=False)
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

        if step == 0:
            return ({'requests':all_requests, 'rides':all_rides}, vot_sample,
                    vot_sample_dict, exmas_params, actual_class_membership)

    if step == 1:
        assert kwargs['run_config'], 'run_config needs to be passed'
        folder = kwargs['run_config']['path_results'] + 'Results/'

        if save_load == 'save':
            assert all(t in kwargs for t in
                       ['class_membership_stability', 'results_daily',
                        'all_results_aggregated', 'predicted_travellers_satisfaction',
                        'actual_travellers_satisfaction']), \
                ('class_membership_stability, results_daily, '
                 'all_results_aggregated, predicted_travellers_satisfaction,'
                 'actual_travellers_satisfaction are required')

            create_directory(kwargs['run_config']['path_results'] + 'Results')

            with open(folder + 'tracked_classes' + '.json', 'w') as _file:
                json.dump(kwargs['class_membership_stability'], _file)
            kwargs['results_daily'].to_csv(folder + 'results_daily' + '.csv', index_label='metric')
            with open(folder + 'all_results_aggregated.pickle', 'wb') as _file:
                pickle.dump(kwargs['all_results_aggregated'], _file)
            with open(folder + 'predicted_travellers_satisfaction.json', 'w') as _file:
                json.dump(kwargs['predicted_travellers_satisfaction'], _file)
            with open(folder + 'actual_travellers_satisfaction.json', 'w') as _file:
                json.dump(kwargs['actual_travellers_satisfaction'], _file)

        if save_load == 'load':
            with (open(folder + 'tracked_classes' + '.json', 'r') as _file):
                class_membership_stability = json.load(_file)
                class_membership_stability = {
                    data_type: {int(pax): {int(cl): prob for cl, prob in probs.items()}
                                for pax, probs in data.items()}
                    for data_type, data in class_membership_stability.items()
                }
            results_daily = pd.read_csv(folder + 'results_daily' + '.csv')
            with open(folder + 'all_results_aggregated.pickle', 'rb') as _file:
                all_results_aggregated = pickle.load(_file)
            with open(folder + 'predicted_travellers_satisfaction.json', 'r') as _file:
                predicted_travellers_satisfaction = json.load(_file)
            with open(folder + 'actual_travellers_satisfaction.json', 'r') as _file:
                actual_travellers_satisfaction = json.load(_file)

            return (vot_sample, exmas_params, actual_class_membership,
                    class_membership_stability, results_daily, all_results_aggregated,
                    predicted_travellers_satisfaction, actual_travellers_satisfaction)

    return None

