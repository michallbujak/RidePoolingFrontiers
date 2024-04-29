import argparse

import pickle

from ExMAS.probabilistic_exmas import main as exmas_algo
import Individual_pricing.pricing_utils.batch_preparation as bt_prep
from Individual_pricing.exmas_loop import exmas_loop_func
from Individual_pricing.matching import matching_function
from Individual_pricing.evaluation import *
from Individual_pricing.pricing_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--directories-json", type=str, required=True)
parser.add_argument("--batch-size", nargs='+', type=int, default=[148, 152])
parser.add_argument("--sample-size", type=int, default=100)
parser.add_argument("--behavioural-size", type=int, default=20)
parser.add_argument("--save-partial", action="store_true")
parser.add_argument("--load-partial", nargs='+', type=int, default=[0, 0, 0])
parser.add_argument("--simulation-name", type=str or None, default=None)
args = parser.parse_args()
print(args)

assert sum(args.load_partial) <= 1, "Cannot load more than 1 intermediate step"

""" Import configuration & prepare results folder"""
directories = bt_prep.get_parameters(args.directories_json)
bt_prep.create_results_directory(directories, args.simulation_name)

if not sum(args.load_partial):
    """ Prepare requests """
    databanks_list, exmas_params = bt_prep.prepare_batches(
        exmas_params=bt_prep.get_parameters(directories.initial_parameters),
        filter_function=lambda x: (len(x.requests) >= args.batch_size[0]) &
                                  (len(x.requests) <= args.batch_size[1])
    )
    # exmas_params = bt_prep.update_probabilistic(directories, exmas_params)

    """ Run the original ExMAS for an initial shareability graph """
    exmas_params.type_of_distribution = None
    databanks_list = exmas_loop_func(
        exmas_algorithm=exmas_algo,
        exmas_params=exmas_params,
        list_databanks=databanks_list
    )

    if args.save_partial:
        with open(directories.partial_results + "initial_exmas_" +
                  str(args.batch_size) + ".pickle", "wb") as file:
            pickle.dump((databanks_list, exmas_params), file)

if args.load_partial[0]:
    with open(directories.partial_results + "initial_exmas_" +
              str(args.batch_size) + ".pickle", "rb") as file:
        databanks_list, exmas_params = pickle.load(file)

if not sum(args.load_partial[1:]):
    """ Extend dataframe rides & prepare behavioural samples """
    databanks_list = [expand_rides(t) for t in databanks_list]
    databanks_list = [prepare_samples(t, args.behavioural_size) for t in databanks_list]

    if args.save_partial:
        with open(directories.partial_results + "expanded_rides_" +
                  str(args.batch_size) + "_" + str(args.sample_size)
                  + ".pickle", "wb") as file:
            pickle.dump((databanks_list, exmas_params), file)

if args.load_partial[1]:
    with open(directories.partial_results + "expanded_rides_" +
              str(args.batch_size) + "_" + str(args.sample_size)
              + ".pickle", "rb") as file:
        databanks_list, exmas_params = pickle.load(file)

if not sum(args.load_partial[2:]):
    """ Calculate new (probabilistic) measures """
    databanks_list = [
        expected_profitability_function(t,
                                        final_sample_size=args.sample_size,
                                        price=exmas_params["price"] / 1000,
                                        speed=exmas_params["avg_speed"],
                                        one_shot=False,
                                        guaranteed_discount=0
                                        )
        for t in databanks_list
    ]

    if args.save_partial:
        with open(directories.partial_results + "calculated_rides_" +
                  str(args.batch_size) + "_" + str(args.sample_size)
                  + ".pickle", "wb") as file:
            pickle.dump((databanks_list, exmas_params), file)

if args.load_partial[2]:
    with open(directories.partial_results + "calculated_rides_" +
              str(args.batch_size) + "_" + str(args.sample_size)
              + ".pickle", "rb") as file:
        databanks_list, exmas_params = pickle.load(file)

""" Conduct matching """
databanks_list = [
    matching_function(
        databank=db,
        params=exmas_params,
        objectives=["profitability"],
        min_max="max"
    ) for db in databanks_list
]

with open(directories.path_results + "results_" + str(args.batch_size) +
          "_" + str(args.sample_size) + ".pickle", "wb") as f:
    pickle.dump(databanks_list, f)

x = 0