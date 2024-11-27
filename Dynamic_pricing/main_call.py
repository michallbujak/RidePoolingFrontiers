""" Wrapper for dynamic pricing algorithm """
import argparse
import ast
import json
import random
import secrets

import pandas as pd
import numpy as np

import Individual_pricing.pricing_utils.batch_preparation as batch_prep
from Individual_pricing.pricing_functions import expand_rides
from Individual_pricing.exmas_loop import exmas_loop_func
from ExMAS.probabilistic_exmas import main as exmas_algo
from Dynamic_pricing.auxiliary_functions import prepare_samples, optimise_discounts

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
compute_save = [args.starting_step == 0, 0, args.starting_step, args.save_partial]

""" Initial data processing """

# Step PP1: Prepare behavioural samples (variable: value of time
if compute_save[0]:
    agents_class_prob = {j: [1/2, 1/2] for j in range(run_config.batch_size)}
    bt_sample = prepare_samples(
        sample_size=run_config.sample_size,
        means=run_config.means,
        st_devs=run_config.st_devs,
        seed=args.seed
    )

# Step PP2: Obtain demand
if compute_save[0]:
    demand, exmas_params = batch_prep.prepare_batches(
        exmas_params=batch_prep.get_parameters(run_config.initial_parameters),
        filter_function=lambda x: len(x.all_requests) == run_config.batch_size,
        quick_load=True,
        batch_size=run_config.batch_size
    )

# Step PP3: Create a dense shareability graph & data manipulation
if compute_save[0]:
    demand = exmas_loop_func(
        exmas_algorithm=exmas_algo,
        exmas_params=exmas_params,
        list_databanks=[demand]
    )
    demand = expand_rides(demand[0])

# Save data if requested
if compute_save[0] & compute_save[3]:
    batch_prep.create_directory(run_config.partial_results + 'Step_0')
    folder = run_config.partial_results + 'Step_0/'
    all_requests = demand['exmas']['requests']
    all_rides = demand['exmas']['rides']
    all_requests.to_csv(folder + 'demand_sample_' + str(run_config.batch_size) + '.csv',
                        index=False)
    all_rides.to_csv(folder + 'rides' + '_' + str(run_config.batch_size) + '.csv', index=False)
    np.save(folder + 'sample' + '_' + str(run_config.sample_size), bt_sample)
    with open(folder + 'exmas_config_' + str(run_config.sample_size) + '.json', 'w') as _file:
        json.dump(exmas_params, _file)

# Skip steps PP1-PP3 and load data
if compute_save[2] - compute_save[1] == 1:
    all_rides, all_requests, bt_sample = None, None, None
    folder = run_config.partial_results + 'Step_0/'
    all_requests = pd.read_csv(folder + 'demand_sample_' + str(run_config.batch_size) + '.csv')
    all_rides = pd.read_csv(folder + 'rides' + '_' + str(run_config.batch_size) + '.csv',
                            converters={k: ast.literal_eval for k in
                                    ['indexes', 'u_paxes', 'individual_times', 'individual_distances']})
    bt_sample = np.load(folder + 'sample' + '_' + str(run_config.sample_size) + '.npy')
    with open(folder + 'exmas_config_' + str(run_config.sample_size) + '.json', 'r') as _file:
        exmas_params = json.load(_file)

compute_save[1] += 1
compute_save[0] = compute_save[2] <= compute_save[1]

""" Proceed to evolutionary part of the analysis """
users_per_day = {}
class_membership_prob = {u: {_: v for _, v in enumerate(run_config['class_probs'])}
                         for u in range(150)}
times_non_shared = dict(all_requests['ttrav'])

for day in range(run_config.no_days):
    # Step E1: filter the shareability graph for a users on a current day
    rng = np.random.default_rng(secrets.randbits(args.seed))
    random.seed(args.seed)
    _ = int(rng.normal(run_config.daily_users, run_config.daily_users/100))
    no_users = _ if (_ > 0 & _ <= run_config.daily_users) else run_config.batch_size
    users = sorted(random.sample(range(run_config.batch_size), no_users))
    users_per_day[day] = users.copy()
    rides_day = all_rides.loc[all_rides['indexes'].apply(
        lambda _x: all(t in users for t in _x)
    )]
    requests_day = all_requests.loc[all_requests['index'].apply(lambda _x: _x in users)]

    # Step E2: Optimal pricing
    rides_day = optimise_discounts(
        rides=rides_day,
        requests=requests_day,
        class_membership=class_membership_prob,
        times_ns=times_non_shared,
        bt_sample=bt_sample,
        bs_levels=[1, 1, 1.1, 1.2, 1.4, 2],
        objective_func=lambda x: x[0] - run_config.mileage_sensitivity*x[4] - run_config.flat_fleet_cost,
        min_acceptance=run_config.minimum_acceptance_probability,
        guaranteed_discount=run_config.guaranteed_discount,
        fare=exmas_params['price'],
        speed=exmas_params['avg_speed']
    )

