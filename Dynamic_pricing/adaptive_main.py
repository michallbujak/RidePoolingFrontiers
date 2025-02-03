""" Wrapper for dynamic pricing algorithm """
import argparse
import ast
import json
import pickle
import random
import secrets
from typing import List, Any

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

import Individual_pricing.pricing_utils.batch_preparation as batch_prep
from Individual_pricing.pricing_utils.batch_preparation import create_directory
from NYC_tools.nyc_data_load import adjust_nyc_request_to_exmas as import_nyc
from Individual_pricing.pricing_functions import expand_rides
from Individual_pricing.exmas_loop import exmas_loop_func
from Individual_pricing.matching import matching_function, matching_function_light
from ExMAS.probabilistic_exmas import main as exmas_algo
from Dynamic_pricing.auxiliary_functions import (prepare_samples, optimise_discounts_future,
                                                 bayesian_vot_updated, aggregate_daily_results,
                                                 check_if_stabilised, all_class_tracking,
                                                 update_satisfaction)

parser = argparse.ArgumentParser()
parser.add_argument("--directories-json", type=str, required=True)
parser.add_argument("--save-partial", action="store_false")
parser.add_argument("--starting-step", type=int, default=0)
parser.add_argument("--simulation-name", type=str or None, default=None)
parser.add_argument("--seed", type=int, default=123)
args = parser.parse_args()
print(args)

# Import configuration & prepare results folder
run_config = batch_prep.get_parameters(args.directories_json)
batch_prep.create_results_directory(
    run_config,
    str(run_config.batch_size) + "_" + str(run_config.sample_size),
    new_directories=False,
    directories_path=args.directories_json
)

# Prepare a save/load variable to run following blocks
computeSave: list[bool | int] = \
    [args.starting_step == 0, 0, args.starting_step, args.save_partial]

""" Step 0: Initial data processing """
# Initialise random generator
global_rng = np.random.default_rng(secrets.randbits(args.seed))

# Step PP1: Prepare behavioural samples (variable: value of time
if computeSave[0]:
    vot_sample = prepare_samples(
        sample_size=run_config.sample_size,
        means=run_config.means,
        st_devs=run_config.st_devs,
        random_state=global_rng
    )
    actual_class_membership = {
        t: global_rng.choice(a=list(range(len(run_config['class_probs']))), size=1,
                     replace=True, p=run_config['class_probs'])[0]
        for t in range(run_config['batch_size'])
    }

# Step PP2: Obtain demand
if computeSave[0]:
    exmas_params = batch_prep.get_parameters(run_config['initial_parameters'])
    demand = import_nyc(
        nyc_requests_path=exmas_params['paths']['requests'],
        skim_matrix_path=exmas_params['paths']['skim'],
        batch_size=run_config['batch_size'],
        start_time=pd.Timestamp(exmas_params['start_time']),
        interval_length_minutes=exmas_params['interval_length_minutes'],
        random_state=global_rng
    )

# Step PP3: Create a dense shareability graph & data manipulation
if computeSave[0]:
    demand = exmas_loop_func(
        exmas_algorithm=exmas_algo,
        exmas_params=exmas_params,
        list_databanks=[demand]
    )
    demand = expand_rides(demand[0])

# Save data if requested
if computeSave[0] & computeSave[3]:
    batch_prep.create_directory(run_config.path_results + 'Step_0')
    folder = run_config.path_results + 'Step_0/'
    all_requests = demand['exmas']['requests']
    all_rides = demand['exmas']['rides']
    all_requests.to_csv(folder + 'demand_sample_' + str(run_config.batch_size) + '.csv',
                        index=False)
    all_rides.to_csv(folder + 'rides' + '_' + str(run_config.batch_size) + '.csv', index=False)
    np.save(folder + 'sample' + '_' + str(run_config.sample_size), vot_sample)
    with open(folder + 'exmas_config.json', 'w') as _file:
        json.dump(exmas_params, _file)
    with open(folder + 'class_memberships.json', 'w') as _file:
        json.dump({k: str(v) for k, v in actual_class_membership.items()}, _file)

# Skip steps PP1-PP3 and load data
if computeSave[2] - computeSave[1] == 1:
    all_rides, all_requests, vot_sample = None, None, None
    folder = run_config.path_results + 'Step_0/'
    all_requests = pd.read_csv(folder + 'demand_sample_' + str(run_config.batch_size) + '.csv')
    all_rides = pd.read_csv(folder + 'rides' + '_' + str(run_config.batch_size) + '.csv',
                            converters={k: ast.literal_eval for k in
                                    ['indexes', 'u_paxes', 'individual_times', 'individual_distances']})
    vot_sample = np.load(folder + 'sample' + '_' + str(run_config.sample_size) + '.npy')
    with open(folder + 'exmas_config.json', 'r') as _file:
        exmas_params = json.load(_file)
    with open(folder + 'class_memberships.json', 'r') as _file:
        actual_class_membership = json.load(_file)
    actual_class_membership = {int(k): int(v) for k, v in actual_class_membership.items()}

computeSave[1] += 1
computeSave[0] = computeSave[2] <= computeSave[1]

""" Step 1: Evolutionary analysis """
if computeSave[0]:
    # A basic filter applied to the shareability set
    all_rides = all_rides.loc[
        [t['u_veh']*exmas_params['avg_speed'] < sum(t['individual_distances'])
         for num, t in all_rides.iterrows()]]

    # Extract features for easier implementation
    times_non_shared = dict(all_requests['ttrav'])
    ns_utilities_all = {pax: utility for pax, utility in zip(all_requests['index'], all_requests['u'])}
    predicted_travellers_satisfaction = {0: {k: run_config['starting_satisfaction']
                                             for k in range(run_config['batch_size'])}}
    actual_travellers_satisfaction = {0: {k: run_config['starting_satisfaction']
                                   for k in range(run_config['batch_size'])}}
    class_membership_prob: dict = {u: {_: v for _, v in enumerate(run_config['class_probs'])}
                             for u in range(run_config['batch_size'])}

    all_results_aggregated = []
    users_per_day = {}
    results_daily = []
    stabilised = []
    last_schedule = [] # for stability analysis

    class_membership_stability = {
        'updated': {ko: {ki: [vi] for ki, vi in vo.items()}
         for ko, vo in class_membership_prob.items()},
        'all': {ko: {ki: [vi] for ki, vi in vo.items()}
         for ko, vo in class_membership_prob.items()}
    }

    progress_bar = tqdm(total=run_config.no_days)
    for day in range(run_config.no_days):
        # Step IP1: filter the shareability graph for a users on a current day
        sampled_values = global_rng.random(run_config['batch_size'])
        users = sorted([pax for num, (pax, sat) in enumerate(actual_travellers_satisfaction[day].items())
                        if np.exp(sat)/(1+np.exp(sat)) > sampled_values[num]])
        users_per_day[day] = users.copy()
        rides_day = all_rides.loc[all_rides['indexes'].apply(
            lambda _x: all(t in users for t in _x)
        )]
        requests_day = all_requests.loc[[t in users for t in all_requests['index']]] # potentially useless

        # Step IP2: Optimal pricing
        rides_day = optimise_discounts_future(
            rides=rides_day,
            class_membership=class_membership_prob,
            times_ns=times_non_shared,
            bt_sample=vot_sample,
            bs_levels=[1, 1, 1.2, 1.3, 1.5, 2],
            travellers_satisfaction=predicted_travellers_satisfaction,
            ns_utilities=ns_utilities_all,
            objective_func=lambda x: x[0] - run_config['mileage_sensitivity']*x[4] -
                                     run_config['flat_fleet_cost'],
            min_acceptance=run_config.minimum_acceptance_probability,
            guaranteed_discount=run_config.guaranteed_discount,
            fare=exmas_params['price'],
            speed=exmas_params['avg_speed'],
            max_discount=run_config['max_discount']
        )

        # Step IP3: Matching
        day_results = matching_function_light(
            _rides=rides_day,
            _requests=requests_day,
            databank={'rides': rides_day, 'requests': requests_day},
            params=exmas_params,
            objectives=['objective'],
            min_max='max',
            filter_rides=False,
            opt_flag='',
            rides_requests=True,
            requestsErrorIndex=True
        )
        schedule_indexes = day_results['schedules']['objective']['indexes']

        # We concluded probabilistic analysis
        # We proceed to sampling decisions and Bayesian estimation

        all_results_aggregated.append(day_results.copy())

        # Step B1: extracting probability
        individualProbability = {}
        sharingSchedule = day_results['schedules']['objective'].copy()
        sharingSchedule = sharingSchedule.loc[[len(t)>1 for t in sharingSchedule['indexes']]]
        sharingSchedule = sharingSchedule.reset_index(inplace=False, drop=True)
        sampledDecisionValues = global_rng.random(size=sum(len(t) for t in sharingSchedule['indexes']))

        # Step B2: update class membership
        predicted_travellers_satisfaction[day+1] = {}
        actual_travellers_satisfaction[day+1] = {}
        sampledDecisions = {}

        decisionValueIndicator: int = 0
        sharingScheduleDecisions = [[]]*len(sharingSchedule)
        updated_travellers = []
        for num, row in sharingSchedule.iterrows():
            for pax, prob, cond_prob in zip(row['indexes'], row['best_profit'][3], row['best_profit'][6]):
                pax_class = actual_class_membership[pax]
                decision = cond_prob[pax_class] > sampledDecisionValues[decisionValueIndicator]
                decisionValueIndicator += 1
                class_membership_prob = bayesian_vot_updated(
                    decision=decision,
                    pax_id=pax,
                    apriori_distribution=class_membership_prob,
                    conditional_probs=cond_prob,
                    distribution_history=class_membership_stability
                )
                sharingScheduleDecisions[num] = sharingScheduleDecisions[num] + [decision]
                updated_travellers += [pax]

            # Update actual satisfaction
            predicted_travellers_satisfaction[day+1], actual_travellers_satisfaction[day+1] \
                = update_satisfaction(
                predicted_travellers_satisfaction_day=predicted_travellers_satisfaction[day+1],
                actual_travellers_satisfaction_day=actual_travellers_satisfaction[day+1],
                rides_row=row,
                predicted_class_distribution=class_membership_prob,
                actual_class_distribution=actual_class_membership,
                predicted_satisfaction=predicted_travellers_satisfaction[day],
                actual_satisfaction=actual_travellers_satisfaction[day],
                vot_sample=vot_sample,
                bs_levels=[1, 1, 1.2, 1.3, 1.5, 2],
                speed=exmas_params['avg_speed'],
                fare=exmas_params['price']
            )

        predicted_travellers_satisfaction[day+1] = (
                predicted_travellers_satisfaction[day].copy() | predicted_travellers_satisfaction[day+1])
        actual_travellers_satisfaction[day+1] = (
                actual_travellers_satisfaction[day].copy() | actual_travellers_satisfaction[day+1])


        # Step IA1: collect data after each run to compare system performance
        class_membership_stability = all_class_tracking(
            class_membership_stability,
            updated_travellers,
            list(range(run_config.batch_size))
        )

        results_daily.append(aggregate_daily_results(
            day_results=day_results,
            decisions=sharingScheduleDecisions,
            fare=exmas_params['price'],
            guaranteed_discount = run_config['guaranteed_discount']
        ))
        # Step IA2: Check if the results stabilised
        last_schedule, stabilised = check_if_stabilised(day_results, last_schedule, stabilised)
        progress_bar.update(1)

    results_daily = pd.concat(results_daily, axis=1)

if computeSave[0] & computeSave[3]:
    batch_prep.create_directory(run_config.path_results + 'Results')
    folder = run_config.path_results + 'Results/'

    with open(folder + 'tracked_classes' + '.json', 'w') as _file:
        json.dump(class_membership_stability, _file)

    results_daily.to_csv(folder + 'results_daily' + '.csv', index_label='metric')

    with open(folder + 'daily_data.pickle', 'wb') as _file:
        pickle.dump(day_results, _file)

# Skip prior and load data
if computeSave[2] - computeSave[1] == 1:
    folder = run_config.path_results

    with open(folder + 'Step_0/' + 'class_memberships' + '.json', 'r') as _file:
        actual_class_membership = json.load(_file)
        actual_class_membership = {int(k): int(v) for k, v in actual_class_membership.items()}
    with (open(folder + 'Results/' + 'tracked_classes' + '.json', 'r') as _file):
        class_membership_stability = json.load(_file)
        class_membership_stability = {
            data_type: {int(pax): {int(cl): prob for cl, prob in probs.items()}
                                    for pax, probs in data.items()}
            for data_type, data in class_membership_stability.items()
        }

    with open(folder + 'Results/' + 'daily_data.pickle', 'rb') as _file:
        day_results = pickle.load(_file)

    results_daily = pd.read_csv(folder + 'Results/' + 'results_daily' + '.csv')

computeSave[1] += 1
computeSave[0] = computeSave[2] <= computeSave[1]

""" Step 2: Post-simulation analysis"""
if computeSave[0]:
    out_path = run_config.path_results + 'Results/figs_tables/'
    create_directory(out_path)

    # Class convergence
    daily_error_pax = {pax: [1 - probs[actual_class_membership[pax]][day] for day in range(len(probs[0]))]
                       for pax, probs in class_membership_stability['updated'].items()}
    error_by_day = [[] for day in range(max(len(err_pax) for err_pax in daily_error_pax.values()))]
    for pax_error in daily_error_pax.values():
        for day, prob in enumerate(pax_error):
            error_by_day[day].append(prob)

    plt.errorbar(x=range(len(error_by_day)),
                 y=[np.mean(day) for day in error_by_day],
                 yerr=[np.std(day) for day in error_by_day])
    plt.savefig(out_path + 'class_error', dpi=args.__dict__.get('dpi', 200))

    # Average class probability
    actual_sampled_freq = {class_id: sum(1 for t in actual_class_membership
                                         if t == class_id)
                      for class_id in range(len(run_config['class_probs']))}
