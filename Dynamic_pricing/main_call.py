""" Wrapper for dynamic pricing algorithm """
import argparse

import Individual_pricing.pricing_utils.batch_preparation as bt_prep
from Individual_pricing.pricing_functions import expand_rides
from Individual_pricing.exmas_loop import exmas_loop_func
from ExMAS.probabilistic_exmas import main as exmas_algo

parser = argparse.ArgumentParser()
parser.add_argument("--directories-json", type=str, required=True)
parser.add_argument("--profitability", action="store_false")
parser.add_argument("--min-accept", type=float, default=0.1)
parser.add_argument("--operating-cost", type=float, default=1)
parser.add_argument("--batch-size", type=int, default=150)
parser.add_argument("--sample-size", type=int, default=5)
parser.add_argument("--save-partial", action="store_false")
parser.add_argument("--load-partial", nargs='+', type=int, default=[0, 0, 0])
parser.add_argument("--simulation-name", type=str or None, default=None)
args = parser.parse_args()
print(args)

assert sum(args.load_partial) <= 1, "Cannot load more than 1 intermediate step"

# Import configuration & prepare results folder
directories = bt_prep.get_parameters(args.directories_json)
bt_prep.create_results_directory(
    directories,
    args.batch_size + "_" + str(args.sample_size),
    new_directories=True,
    directories_path=args.directories_json
)

# Prepare a save/load variable to run following blocks
save_load = [0, args.load_partial, args.save_partial]

# Step 1: Prepare behavioural samples
agents_class_prob = {j: [1/2, 1/2] for j in range(args.batch_size)}


# Step 1: Obtain demand
if save_load[1][save_load[0]]:
    demand, exmas_params = bt_prep.prepare_batches(
        exmas_params=bt_prep.get_parameters(directories.initial_parameters),
        filter_function=lambda x: len(x.requests) == args.batch_size[0]
    )

# Step 2: Create a dense shareability graph & data manipulation
if save_load[1][save_load[0]]:
    demand = exmas_loop_func(
        exmas_algorithm=exmas_algo,
        exmas_params=exmas_params,
        list_databanks=demand
    )
    demand = expand_rides(demand[0])
