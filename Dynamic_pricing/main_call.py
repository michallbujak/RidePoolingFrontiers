""" Wrapper for dynamic pricing algorithm """
import argparse
import ast
import random
import secrets

import pandas as pd
import numpy as np

import Individual_pricing.pricing_utils.batch_preparation as bt_prep
from Individual_pricing.pricing_functions import expand_rides
from Individual_pricing.exmas_loop import exmas_loop_func
from ExMAS.probabilistic_exmas import main as exmas_algo
from Dynamic_pricing.auxiliary_functions import prepare_samples

parser = argparse.ArgumentParser()
parser.add_argument("--directories-json", type=str, required=True)
parser.add_argument("--profitability", action="store_true")
parser.add_argument("--days", type=int, default=5)
parser.add_argument("--daily-users", type=int, default=100)
parser.add_argument("--min-accept", type=float, default=0.1)
parser.add_argument("--operating-cost", type=float, default=1)
parser.add_argument("--batch-size", type=int, default=150)
parser.add_argument("--sample-size", type=int, default=5)
parser.add_argument("--save-partial", action="store_false")
parser.add_argument("--starting-step", type=int, default=0)
parser.add_argument("--simulation-name", type=str or None, default=None)
parser.add_argument("--seed", type=int, default=123)
args = parser.parse_args()
print(args)

# Import configuration & prepare results folder
directories = bt_prep.get_parameters(args.directories_json)
bt_prep.create_results_directory(
    directories,
    str(args.batch_size) + "_" + str(args.sample_size),
    new_directories=True,
    directories_path=args.directories_json
)

# Prepare a save/load variable to run following blocks
compute_save = [args.starting_step == 0, 0, args.starting_step, args.save_partial]

""" Initial data processing """

# Step PP1: Prepare behavioural samples (variable: value of time
if compute_save[0]:
    agents_class_prob = {j: [1/2, 1/2] for j in range(args.batch_size)}
    sample = prepare_samples(
        sample_size=args.sample_size,
        means=directories.means,
        st_devs=directories.st_devs,
        seed=args.seed
    )

# Step PP2: Obtain demand
if compute_save[0]:
    demand, exmas_params = bt_prep.prepare_batches(
        exmas_params=bt_prep.get_parameters(directories.initial_parameters),
        filter_function=lambda x: len(x.requests) == args.batch_size,
        quick_load=True,
        batch_size=args.batch_size
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
    bt_prep.create_directory(directories.partial_results + 'Step_0')
    folder = directories.partial_results + '/Step_0/'
    requests = demand['exmas']['requests']
    rides = demand['exmas']['rides']
    requests.to_csv(folder + 'demand_sample_' + '_' + str(args.batch_size) + '.csv',
                    index=False)
    rides.to_csv(folder + 'rides' + '_' + str(args.batch_size) + '.csv', index=False)
    np.save(folder + 'sample' + '_' + str(args.sample_size), sample)

# Skip steps PP1-PP3 and load data
if compute_save[2] - compute_save[1] == 1:
    rides, requests, sample = None, None, None
    folder = directories.partial_results + 'Step_0/'
    requests = pd.read_csv(folder + 'demand_sample_' + '_' + str(args.batch_size) + '.csv')
    rides = pd.read_csv(folder + 'rides' + '_' + str(args.batch_size) + '.csv',
                        converters={k: ast.literal_eval for k in
                                    ['indexes', 'u_paxes', 'individual_times', 'individual_distances']})
    sample = np.load(folder + 'sample' + '_' + str(args.sample_size) + '.npy')

compute_save[1] += 1
compute_save[0] = compute_save[2] <= compute_save[1]

""" Proceed to evolutionary part of the analysis """
users_per_day = {}
for day in range(args.days):
    # Step E1: filter the shareability graph for a users on a current day
    rng = np.random.default_rng(secrets.randbits(args.seed))
    random.seed(args.seed)
    _ = int(rng.normal(args.daily_users, args.daily_users/100))
    no_users = _ if (_ > 0 & _ <= args.daily_users) else args.batch_size
    users = sorted(random.sample(range(150), no_users))
    users_per_day[day] = users.copy()
    rides_day = rides.loc[rides['indexes'].apply(
        lambda _x: all(t in users for t in _x)
    )]
    requests_day = requests.loc[requests['index'].apply(lambda _x: _x in users)]

    # Step E2: Optimal pricing
