""" Wrapper for dynamic pricing algorithm """
import argparse

import Individual_pricing.pricing_utils.batch_preparation as bt_prep
from Individual_pricing.pricing_functions import expand_rides
from Individual_pricing.exmas_loop import exmas_loop_func
from ExMAS.probabilistic_exmas import main as exmas_algo
from Dynamic_pricing.auxiliary_functions import prepare_samples

parser = argparse.ArgumentParser()
parser.add_argument("--directories-json", type=str, required=True)
parser.add_argument("--profitability", action="store_false")
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

# Step 1: Prepare behavioural samples (variable: value of time
if compute_save[0]:
    agents_class_prob = {j: [1/2, 1/2] for j in range(args.batch_size)}
    sample = prepare_samples(
        sample_size=args.sample_size,
        means=directories.means,
        st_devs=directories.st_devs,
        seed=args.seed
    )

# Step 2: Obtain demand
if compute_save[0]:
    demand, exmas_params = bt_prep.prepare_batches(
        exmas_params=bt_prep.get_parameters(directories.initial_parameters),
        filter_function=lambda x: len(x.requests) == args.batch_size
    )

# Step 3: Create a dense shareability graph & data manipulation
if compute_save[0]:
    demand = exmas_loop_func(
        exmas_algorithm=exmas_algo,
        exmas_params=exmas_params,
        list_databanks=demand
    )
    demand_full = expand_rides(demand[0])

if compute_save[0] & compute_save[3]:
    bt_prep.create_directory('Step_0')
    folder = directories.partial_results + '/Step_0/'
    demand_name = 'demand_sample_' + str(args.batch_size) + '_' + str(args.sample_size)
    skim_name = 'skim'
    demand = demand_full['requests']
    skim = demand_full['skim']
    demand.to_csv(folder + demand_name + '.csv')
    skim.to_csv(folder + skim_name + '.csv')



if compute_save[2] - compute_save[1] == 1:
    demand, sample = 'LOAD' #TODO

compute_save[1] += 1
compute_save[0] = compute_save[2] <= compute_save[1]

""" Proceed to evolutionary estimation """

