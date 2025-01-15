""" Wrapper for dynamic pricing algorithm """
import argparse
import ast
import json
import random
import secrets
from typing import List, Any

import pandas as pd
import numpy as np
from tqdm import tqdm

import Individual_pricing.pricing_utils.batch_preparation as batch_prep
from NYC_tools.nyc_data_load import adjust_nyc_request_to_exmas as import_nyc
from Individual_pricing.pricing_functions import expand_rides
from Individual_pricing.exmas_loop import exmas_loop_func
from Individual_pricing.matching import matching_function
from ExMAS.probabilistic_exmas import main as exmas_algo
from Dynamic_pricing.auxiliary_functions import (prepare_samples, optimise_discounts,
                                                 bayesian_vot_updated, aggregate_daily_results)

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

""" Initial data processing """
# Step 0: Initialise random generator
global_rng = np.random.default_rng(secrets.randbits(args.seed))

# Step PP1: Prepare behavioural samples (variable: value of time
if computeSave[0]:
    votSample = prepare_samples(
        sample_size=run_config.sample_size,
        means=run_config.means,
        st_devs=run_config.st_devs,
        random_state=global_rng
    )
    actualClassMembership = {
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
    allRequests = demand['exmas']['requests']
    allRides = demand['exmas']['rides']
    allRequests.to_csv(folder + 'demand_sample_' + str(run_config.batch_size) + '.csv',
                       index=False)
    allRides.to_csv(folder + 'rides' + '_' + str(run_config.batch_size) + '.csv', index=False)
    np.save(folder + 'sample' + '_' + str(run_config.sample_size), votSample)
    with open(folder + 'exmas_config_' + str(run_config.sample_size) + '.json', 'w') as _file:
        json.dump(exmas_params, _file)
    with open(folder + 'class_memberships_' + str(run_config.sample_size) + '.json', 'w') as _file:
        json.dump({k: str(v) for k, v in actualClassMembership.items()}, _file)

# Skip steps PP1-PP3 and load data
if computeSave[2] - computeSave[1] == 1:
    allRides, allRequests, votSample = None, None, None
    folder = run_config.path_results + 'Step_0/'
    allRequests = pd.read_csv(folder + 'demand_sample_' + str(run_config.batch_size) + '.csv')
    allRides = pd.read_csv(folder + 'rides' + '_' + str(run_config.batch_size) + '.csv',
                           converters={k: ast.literal_eval for k in
                                    ['indexes', 'u_paxes', 'individual_times', 'individual_distances']})
    votSample = np.load(folder + 'sample' + '_' + str(run_config.sample_size) + '.npy')
    with open(folder + 'exmas_config_' + str(run_config.sample_size) + '.json', 'r') as _file:
        exmas_params = json.load(_file)
    with open(folder + 'class_memberships_' + str(run_config.sample_size) + '.json', 'r') as _file:
        actualClassMembership = json.load(_file)
    actualClassMembership = {int(k): int(v) for k, v in actualClassMembership.items()}

computeSave[1] += 1
computeSave[0] = computeSave[2] <= computeSave[1]

""" Proceed to evolutionary part of the analysis """

if computeSave[0]:
    users_per_day = {}
    class_membership_prob: dict = {u: {_: v for _, v in enumerate(run_config['class_probs'])}
                             for u in range(run_config['batch_size'])}
    times_non_shared = dict(allRequests['ttrav'])
    resultsDaily = []

    classMembershipStability = {ko: {ki: [vi] for ki, vi in vo.items()}
                                for ko, vo in class_membership_prob.items()}

    progress_bar = tqdm(total=run_config.no_days)
    for day in range(run_config.no_days):
        # Step IP1: filter the shareability graph for a users on a current day
        _ = int(global_rng.normal(run_config.daily_users, run_config.daily_users/100))
        no_users = _ if (_ > 0 & _ <= run_config.daily_users) else run_config.batch_size
        users = sorted(global_rng.choice(range(run_config.batch_size), no_users))
        users_per_day[day] = users.copy()
        rides_day = allRides.loc[allRides['indexes'].apply(
            lambda _x: all(t in users for t in _x)
        )]
        requests_day = allRequests.loc[allRequests['index'].apply(lambda _x: _x in users)] # potentially useless

        # Step IP2: Optimal pricing
        rides_day = optimise_discounts(
            rides=rides_day,
            class_membership=class_membership_prob,
            times_ns=times_non_shared,
            bt_sample=votSample,
            bs_levels=[1, 1, 1.1, 1.2, 1.4, 2],
            objective_func=lambda x: x[0] - run_config['mileage_sensitivity']*x[4] -
                                     run_config['flat_fleet_cost'],
            min_acceptance=run_config.minimum_acceptance_probability,
            guaranteed_discount=run_config.guaranteed_discount,
            fare=exmas_params['price'],
            speed=exmas_params['avg_speed'],
            max_discount=run_config['max_discount']
        )

        # Step IP3: Matching
        dayResults = matching_function(
            databank={'rides': rides_day, 'requests': requests_day},
            params=exmas_params,
            objectives=['objective'],
            min_max='max',
            filter_rides=False,
            opt_flag='',
            rides_requests=True,
            requestsErrorIndex=True
        )
        # We concluded probabilistic analysis
        # We proceed to sampling decisions and Bayesian estimation

        # Step B1: extracting probability
        individualProbability = {}
        sharingSchedule = dayResults['schedules']['objective'].copy()
        sharingSchedule = sharingSchedule.loc[[len(t)>1 for t in sharingSchedule['indexes']]]
        sharingSchedule = sharingSchedule.reset_index(inplace=False, drop=True)
        sampledDecisionValues = global_rng.random(size=sum(len(t) for t in sharingSchedule['indexes']))

        # Step B2: update class membership
        sampledDecisions = {}
        decisionValueIndicator: int = 0
        sharingScheduleDecisions = [[]]*len(sharingSchedule)
        for num, row in sharingSchedule.iterrows():
            for pax, prob, cond_prob in zip(row['indexes'], row['best_profit'][3], row['best_profit'][-2]):
                pax_class = actualClassMembership[pax]
                decision = cond_prob[pax_class] > sampledDecisionValues[decisionValueIndicator]
                decisionValueIndicator += 1
                class_membership_prob = bayesian_vot_updated(
                    decision=decision,
                    pax_id=pax,
                    apriori_distribution=class_membership_prob,
                    conditional_probs=cond_prob,
                    distribution_history=classMembershipStability
                )
                sharingScheduleDecisions[num] = sharingScheduleDecisions[num] + [decision]

        # Step IA1: collect data after each run to compare system performance
        resultsDaily.append(aggregate_daily_results(
            day_results=dayResults,
            decisions=sharingScheduleDecisions,
            fare=exmas_params['price'],
            guaranteed_discount = run_config['guaranteed_discount']
        ))
        progress_bar.update(1)

    resultsDaily = pd.concat(resultsDaily, axis=1)

if computeSave[0] & computeSave[3]:
    batch_prep.create_directory(run_config.path_results + 'Results')
    folder = run_config.path_results + 'Results/'

    with open(folder + 'tracked_classes' + '.json', 'w') as _file:
        json.dump(classMembershipStability, _file)

    resultsDaily.to_csv(folder + 'results_daily' + '.csv')

# Skip prior and load data
if computeSave[2] - computeSave[1] == 1:
    folder = run_config.path_results

    with open(folder + 'Step_0/' + 'actual_classes' + '.json', 'r') as _file:
        actualClassMembership = json.load(_file)
    with open(folder + 'Results/' + 'tracked_classes' + '.json', 'r') as _file:
        classMembershipStability = json.load(_file)

    resultsDaily = pd.read_csv(folder + 'results_daily' + '.csv')

computeSave[1] += 1
computeSave[0] = computeSave[2] <= computeSave[1]
