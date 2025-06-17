""" In progress: cleaned version of the adaptive pricing algorithm """
import argparse
import json
import pickle
import secrets

import pandas as pd
import numpy as np
from tqdm import tqdm

import Individual_pricing.pricing_utils.batch_preparation as batch_prep
from NYC_tools.nyc_data_load import adjust_nyc_request_to_exmas as import_nyc
from Individual_pricing.pricing_functions import expand_rides
from Individual_pricing.exmas_loop import exmas_loop_func
from Individual_pricing.matching import matching_function_light
from ExMAS.probabilistic_exmas import main as exmas_algo
from Dynamic_pricing.save_load_functions import save_load_data
from Dynamic_pricing.auxiliary_functions import (prepare_samples, optimise_discounts_future,
                                                 bayesian_vot_updated, aggregate_daily_results,
                                                 check_if_stabilised, all_class_tracking,
                                                 update_satisfaction, post_run_analysis, benchmarks)

parser = argparse.ArgumentParser()
parser.add_argument("--directories-json", type=str, required=True)
parser.add_argument("--save-partial", action="store_false")
parser.add_argument("--starting-step", type=int, default=0)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--plot-format", type=str, default="png")
parser.add_argument("--plot-dpi", type=int, default=200)
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
compute_save = {
    'starting_step': args.starting_step,
    'save_partial': args.save_partial
}

# Initialise random generator
global_rng = np.random.default_rng(secrets.randbits(args.seed))


""" Data preparation, shareability set """
if compute_save['starting_step'] == 0:
    # Step PP1: Prepare behavioural samples (variable: value of time
    vot_sample = prepare_samples(
        sample_size=run_config.sample_size,
        means=run_config.means,
        st_devs=run_config.st_devs,
        random_state=global_rng
    )

    vot_sample_dict = {k: [t[0] for t in vot_sample if t[1] == k]
                       for k in set(vot_sample[:, 1])}

    actual_class_membership = {
        t: global_rng.choice(a=list(range(len(run_config['class_probs']))), size=1,
                     replace=True, p=run_config['class_probs'])[0]
        for t in range(run_config['batch_size'])
    }

    # Step PP2: Obtain demand
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
    demand = exmas_loop_func(
        exmas_algorithm=exmas_algo,
        exmas_params=exmas_params,
        list_databanks=[demand]
    )

    demand = expand_rides(demand[0])
    demand = demand['exmas']

    if compute_save['save_partial']:
        save_load_data(
            step=0,
            save_load='save',
            run_config=run_config,
            vot_sample=vot_sample,
            demand=demand,
            exmas_params=exmas_params,
            actual_class_membership=actual_class_membership
        )

""" Evolutionary analysis """
if compute_save['starting_step'] == 1:
    demand, vot_sample, vot_sample_dict, exmas_params, actual_class_membership = save_load_data(
        step=0,
        save_load='load',
        run_config=run_config
    )

if compute_save['starting_step'] <= 1:
    # A basic filter applied to the shareability set
    all_rides = demand['rides'].loc[
        [t['u_veh']*exmas_params['avg_speed'] < sum(t['individual_distances']) + 5 # 5 is a buffer for wrong rounding
         for num, t in demand['rides'].iterrows()]]

    # Extract features for easier implementation
    predicted_travellers_satisfaction = {0: {k: run_config['starting_satisfaction']
                                             for k in range(run_config['batch_size'])}}
    actual_travellers_satisfaction = {0: {k: run_config['starting_satisfaction']
                                   for k in range(run_config['batch_size'])}}
    class_membership_prob: dict = {u: {_: v for _, v in enumerate(run_config['class_probs'])}
                             for u in range(run_config['batch_size'])}

    # Variables for tracking
    all_results_aggregated = []
    users_per_day = {}
    results_daily = []
    stabilised = []
    last_schedule = []
    all_sampled_decisions = {}

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
        requests_day = demand['requests'].loc[[t in users
                                               for t in demand['requests']['index']]] # potentially useless

        # Step IP2: Optimal pricing
        rides_day = optimise_discounts_future(
            rides=rides_day,
            class_membership=class_membership_prob,
            vot_sample=vot_sample,
            bs_levels=run_config['pfs_levels'],
            travellers_satisfaction=predicted_travellers_satisfaction,
            objective_func=lambda x: x[0] - run_config['mileage_sensitivity']*x[4] -
                                     run_config['flat_fleet_cost']*x[6],
            min_acceptance=run_config.minimum_acceptance_probability,
            guaranteed_discount=run_config.guaranteed_discount,
            fare=exmas_params['price'],
            speed=exmas_params['avg_speed'],
            max_discount=run_config['max_discount'],
            attraction_sensitivity=run_config['attraction_sensitivity']
        )

        # Step IP3: Matching
        day_results = matching_function_light(
            _rides=rides_day,
            _requests=requests_day,
            _objective='objective',
            _min_max='max',
            rrs_output=True
        )

        # We concluded probabilistic analysis
        # We proceed to sampling decisions and Bayesian estimation

        # Step B1: Sample decisions
        day_results['schedules']['objective']['sampled_vot'] = (
            day_results['schedules']['objective']['indexes'].apply(
                lambda x: [global_rng.choice(vot_sample_dict[actual_class_membership[t]]) for t in x]
        ))
        day_results['schedules']['objective']['decisions'] = (
            day_results['schedules']['objective'].apply(
                lambda x: [x['sampled_vot'][t] <= x['best_profit'][8][t] + 0.001
                           for t in range(len(x['indexes']))],
                axis=1
            )
        )
        day_results['schedules']['objective']['decision'] = (
            day_results['schedules']['objective']['decisions'].apply(all))

        all_results_aggregated.append(day_results.copy())

        sharingSchedule = day_results['schedules']['objective'].copy()
        sharingSchedule = sharingSchedule.loc[[len(t)>1 for t in sharingSchedule['indexes']]]
        sharingSchedule = sharingSchedule.reset_index(inplace=False, drop=True)

        # Step B2: update class membership
        predicted_travellers_satisfaction[day+1] = {}
        actual_travellers_satisfaction[day+1] = {}

        updated_travellers = []
        for num, row in sharingSchedule.iterrows():
            for pax, prob, cond_prob, decision in (
                    zip(row['indexes'], row['best_profit'][3],
                        row['best_profit'][7], row['decisions'])):
                pax_class = actual_class_membership[pax]
                class_membership_prob = bayesian_vot_updated(
                    decision=decision,
                    pax_id=pax,
                    apriori_distribution=class_membership_prob,
                    conditional_probs=cond_prob,
                    distribution_history=class_membership_stability
                )
                updated_travellers += [pax]

            # Update actual satisfaction
            predicted_travellers_satisfaction[day+1], actual_travellers_satisfaction[day+1] \
                = update_satisfaction(
                predicted_travellers_satisfaction_day=predicted_travellers_satisfaction[day+1],
                actual_travellers_satisfaction_day=actual_travellers_satisfaction[day+1],
                schedule_row=row,
                predicted_class_distribution=class_membership_prob,
                predicted_satisfaction=predicted_travellers_satisfaction[day],
                actual_satisfaction=actual_travellers_satisfaction[day],
                vot_sample=vot_sample,
                bs_levels=run_config['pfs_levels'],
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
            predicted_satisfaction=predicted_travellers_satisfaction[day+1],
            actual_satisfaction=actual_travellers_satisfaction[day+1],
            fare=exmas_params['price'],
            run_config = run_config
        ))
        # Step IA2: Check if the results stabilised
        last_schedule, stabilised = check_if_stabilised(day_results, last_schedule, stabilised)
        progress_bar.update(1)

    results_daily = pd.concat(results_daily, axis=1)

    if compute_save['save_partial']:
        save_load_data(
            step=1,
            save_load='save',
            run_config=run_config,
            class_membership_stability=class_membership_stability,
            results_daily=results_daily,
            all_results_aggregated=all_results_aggregated,
            predicted_travellers_satisfaction=predicted_travellers_satisfaction,
            actual_travellers_satisfaction=actual_travellers_satisfaction
        )

if compute_save['starting_step'] == 2:
    (vot_sample, exmas_params, actual_class_membership,
     class_membership_stability, results_daily, all_results_aggregated,
     predicted_travellers_satisfaction, actual_travellers_satisfaction) = save_load_data(
        step=1,
        save_load='load',
        run_config=run_config
    )

""" Step 2: Post-simulation analysis"""
if compute_save['starting_step'] <= 2:
    out_path = run_config.path_results + 'Results/figs_tables/'
    batch_prep.create_directory(out_path)

    post_run_analysis(
        class_membership_stability=class_membership_stability,
        actual_class_membership=actual_class_membership,
        results_daily=results_daily,
        all_results_aggregated=all_results_aggregated,
        predicted_travellers_satisfaction=predicted_travellers_satisfaction,
        actual_travellers_satisfaction=actual_travellers_satisfaction,
        run_config=run_config,
        exmas_params=exmas_params,
        out_path=out_path,
        args=args,
        x_ticks=[t - 1 for t in [1, 5, 10, 15, 20]],
        x_ticks_labels=[str(t) for t in [1, 5, 10, 15, 20]]
    )

    run_config.update(exmas_params)

    benchmarks(
        all_results_aggregated=all_results_aggregated,
        _results_daily=results_daily,
        _actual_satisfaction=actual_travellers_satisfaction,
        _actual_classes=actual_class_membership,
        class_membership_stability=class_membership_stability,
        _run_config=run_config,
        _flat_discount=0.2
    )

