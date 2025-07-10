import argparse
import os

import pickle

from ExMAS.probabilistic_exmas import main as exmas_algo
import Individual_pricing.pricing_utils.batch_preparation as batch_prep
from Individual_pricing.matching import matching_function
from Individual_pricing.pricing_functions import *
from NYC_tools.nyc_data_load import adjust_nyc_request_to_exmas as import_nyc

parser = argparse.ArgumentParser()
parser.add_argument("--directories-json", type=str, required=True)
parser.add_argument("--profitability", action="store_false")
parser.add_argument("--min-accept", type=float, default=0.1)
parser.add_argument("--operating-cost", type=float, default=0.5)
# parser.add_argument("--batch-size", nargs='+', type=int, default=[100, 200])
parser.add_argument("--batch-size", type=int, default=150)
parser.add_argument("--sample-size", type=int, default=5)
parser.add_argument("--save-partial", action="store_false")
parser.add_argument("--load-partial", nargs='+', type=int, default=[0, 0, 0])
parser.add_argument("--simulation-name", type=str or None, default=None)
args = parser.parse_args()
print(args)

assert sum(args.load_partial) <= 1, "Cannot load more than 1 intermediate step"

""" Import configuration & prepare results folder"""
directories = batch_prep.get_parameters(args.directories_json)

if not sum(args.load_partial):
    from Individual_pricing.exmas_loop import exmas_loop_func
    """ Prepare requests """
    # databanks_list, exmas_params = batch_prep.prepare_batches(
    #     exmas_params=batch_prep.get_parameters(directories.initial_parameters),
    #     filter_function=lambda x: (len(x.requests) >= args.batch_size[0]) &
    #                               (len(x.requests) <= args.batch_size[1])
    # )
    exmas_params = batch_prep.get_parameters(directories['initial_parameters'])
    demand = import_nyc(
        nyc_requests_path=exmas_params['paths']['requests'],
        skim_matrix_path=exmas_params['paths']['skim'],
        batch_size=args.batch_size,
        start_time=pd.Timestamp(exmas_params['start_time']),
        interval_length_minutes=exmas_params['interval_length_minutes'],
    )
    databanks_list = [demand]

    """ Run the original ExMAS for an initial shareability graph """
    exmas_params.type_of_distribution = None
    databanks_list = exmas_loop_func(
        exmas_algorithm=exmas_algo,
        exmas_params=exmas_params,
        list_databanks=databanks_list
    )

    batch_prep.create_results_directory(
        directories,
        str(len(databanks_list[0]['requests'])) + "_" + str(args.sample_size),
        new_directories=True,
        directories_path=args.directories_json
    )

    if args.save_partial:
        with open(directories.partial_results + "initial_exmas_" +
                  str(args.batch_size) + ".pickle", "wb") as file:
            pickle.dump((databanks_list, exmas_params), file)

if sum(args.load_partial) >= 1:
    directories.partial_results = (directories.partial_results
                                   + str(args.batch_size)
                                   + "_" + str(args.sample_size) + '/')

if args.load_partial[0]:
    with open(directories.partial_results + "initial_exmas_" +
              str(args.batch_size) + ".pickle", "rb") as file:
        databanks_list, exmas_params = pickle.load(file)

if not sum(args.load_partial[1:]):
    """ Extend dataframe rides & prepare behavioural samples """
    databanks_list = [expand_rides(t) for t in databanks_list]
    databanks_list = [prepare_samples(t, args.sample_size) for t in databanks_list]

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
    if args.profitability:
        func = lambda x: x[0]/x[4] if x[4] != 0 else 0
    else:
        func = lambda x: x[0] - x[4]*args.operating_cost

    databanks_list = [
        expected_profitability_function(t,
                                        max_func=func,
                                        final_sample_size=args.sample_size,
                                        fare=exmas_params["price"],
                                        speed=exmas_params["avg_speed"],
                                        one_shot=False,
                                        guaranteed_discount=0.05,
                                        min_acceptance=args.min_accept
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

databanks_list = [
    profitability_measures(
        databank=t,
        op_costs=[args.operating_cost]
    ) for t in databanks_list
]

""" Conduct matching """
databanks_list = [
    matching_function(
        databank=db,
        params=exmas_params,
        objectives=None,
        min_max="max"
    ) for db in databanks_list
]

if args.profitability:
    with open(directories.path_results + "results_" + str(args.batch_size) +
              "_" + str(args.sample_size) + ".pickle", "wb") as f:
        pickle.dump(databanks_list, f)
else:
    with open(directories.path_results + "results_" + str(args.batch_size) +
              "_" + str(args.sample_size) + "_" + str(args.operating_cost) +
              "_" + str(args.min_accept) + ".pickle", "wb") as f:
        pickle.dump(databanks_list, f)
