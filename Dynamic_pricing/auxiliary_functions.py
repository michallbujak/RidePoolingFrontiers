""" Function to support the dynamic pricing algorithm """
import argparse
import secrets
from typing import Callable, Tuple, Any
import itertools
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple, HandlerBase

from Individual_pricing.matching import matching_function_light


def prepare_samples(
        sample_size: int,
        means: list or tuple,
        st_devs: list or tuple,
        random_state: np.random._generator.Generator or None = None,
        descending: bool = False,
        **kwargs
):
    """
    Prepare behavioural samples to create a discrete distribution
    instead of the continuous normal
    :param sample_size: numer of samples per class
    :param means: means per each class
    :param st_devs: respective standard deviations
    :param random_state: pass existing random state for reproducibility
    :param descending: return values in descending order
    :param kwargs: additional parameters if required
    :return: discrete behavioural samples
    """
    if random_state is None:
        random_state = np.random.default_rng(secrets.randbits(kwargs.get('seed', 123)))
    out = []

    for subpop_num, mean in enumerate(means):
        pop_sample = random_state.normal(
            loc=mean,
            scale=st_devs[subpop_num],
            size=sample_size
        )
        pop_sample = [(t, subpop_num) for t in pop_sample]
        out += pop_sample

    out.sort(reverse=descending)
    return np.array(out)


def row_acceptable_discount_bayes(
        _rides_row: pd.Series,
        _class_membership: dict,
        _bs_levels: dict or list,
        _vot_sample: list[list] or list[tuple],
        _speed: int or float,
        _fare: float
) -> list:
    """
    Samples discounts for which clients accept rides
    at different discount levels
    @param _rides_row: the function works on rows of pd.Dataframe
    @param _class_membership: dictionary with membership probability
    for each traveller for each behavioural class
    @param _bs_levels: penalty for sharing: subject to the number of co-travellers
    @param _vot_sample: sampled value of time across the population
    @param _speed: speed for the individual time calculations
    @param _fare: price per kilometre
    @return: acceptable discount levels (progressive with probability)
    """
    assert 0.1 < _fare < 10, ValueError('The fare should be per kilometre,'
                                        'assumed to be in range (0.1, 10)')
    assert all([1 < t[0] < 100 for t in _vot_sample]), ValueError('VoT should be per hour,'
                                                                 'assumed range (1, 100)')
    assert all([0.5 < t < 10 for t in _bs_levels]), ValueError('PfS/WtS assumed range (1, 100)')

    no_travellers = len(_rides_row["indexes"])
    if no_travellers == 1:
        return []

    out = []

    for no, trav in enumerate(_rides_row["indexes"]):
        pax_out = _rides_row["individual_times"][no]
        pax_out *= _bs_levels[len(_rides_row["indexes"])]
        pax_out -= _rides_row["individual_distances"][no]/_speed
        pax_out = [(t[0]/3600 * pax_out, t[1], t[0]) for t in _vot_sample]
        pax_out = [(t[0], t[1], t[2]) if t[0] >= 0 else (0, t[1], t[2]) for t in pax_out]
        pax_out = [(t[0]/(_fare * _rides_row["individual_distances"][no]/1000), t[1], t[2]) for t in pax_out]
        out.append(pax_out)

    return out

def row_maximise_profit_bayes(
        _rides_row: pd.Series,
        _class_membership: dict[dict],
        _max_output_func: Callable[[list], float],
        _sample_size: int,
        _fare: float = 0.0015,
        _probability_single: float = 1,
        _guaranteed_discount: float = 0.05,
        _min_acceptance: float = 0,
        _max_discount: float = 0.5
):
    """
    Function to calculate the expected performance (or its derivatives)
     of a precalculated exmas ride
    --------
    @param _rides_row: row of exmas rides (potential combination + characteristics)
    @param _class_membership: probability for each agent to belong to a given class
    @param _sample_size: number of samples of value of time for a class
    @param _fare: per-kilometre fare
    @param _probability_single: probability that a customer is satisfied with a private
    ride if offered one
    @param _guaranteed_discount: when traveller accepts a shared ride and any of the co-travellers
    reject a ride, the traveller is offered
    @param _max_output_func: specify what is the maximisation objective
    @param _min_acceptance: minimum acceptance probability
    @param _max_discount: maximum allowed sharing discount
    --------
    @return vector comprising 6 main characteristics when applied discount maximising
    the expected revenue:
    - expected revenue
    - vector of individual discounts
    - revenue from the shared ride if accepted
    - vector of probabilities that individuals accept the shared ride
    - expected distance
    - probability of acceptance when in certain class: [t1 in C1, t1 in C2,...], [t2 in C1, t2 in C2, ...]
    - max output function (by default, profitability)
    """
    if _fare > 1:
        _mFare: float = _fare/1000
    else:
        _mFare: float = _fare

    no_travellers = len(_rides_row["indexes"])
    if no_travellers == 1:
        out = [_probability_single * _rides_row["veh_dist"] * _mFare * (1 - _guaranteed_discount),
               0,
               _rides_row["veh_dist"] * _mFare * (1 - _guaranteed_discount),
               [_probability_single],
               _rides_row["veh_dist"] / 1000 * _probability_single,
               [1]*len(_class_membership[0])
               ]
        out += [_max_output_func(out)]
        return out

    travellers = _rides_row["indexes"]
    discounts = _rides_row["accepted_discount"].copy()
    discounts = [[t if t[0] > _guaranteed_discount else (_guaranteed_discount, t[1])
                 for t in d] for d in discounts] # at least guaranteed discount
    discounts_indexes = list(itertools.product(*[range(len(t)) for t in discounts]))
    # discounts = list(itertools.product(*discounts)) # create a list of bi-vectors (disc, class) of discounts (everyone gets one)
    base_revenues = {num: _rides_row["individual_distances"][num] * _mFare for num, t in
                     enumerate(_rides_row['indexes'])}
    best = [0, 0, 0, 0, 0, 0, 0]

    discount_indexes: tuple[int]
    for discount_indexes in discounts_indexes:
        # Start with the effectively shared ride
        discount = [discounts[_num][_t] for _num, _t in enumerate(discount_indexes)]
        if any(t[0] > _max_discount for t in discount):
            continue
        eff_price = [_mFare * (1 - t[0]) for t in discount]
        revenue_shared = [a * b for a, b in
                          zip(_rides_row["individual_distances"], eff_price)]
        probability_shared = 1
        prob_individual = [.0] * len(discount)

        for num, pax in enumerate(travellers):
            prob_individual[num] = (sum(_class_membership[pax][t[1]]
                                  for t in discounts[num][:(discount_indexes[num]+1)])/
                               _sample_size)
            probability_shared *= prob_individual[num]

        if probability_shared < _min_acceptance:
            continue

        remaining_revenue = 0
        for num_trav in range(len(discount)):
            # First, if the P(X_j = 0)*r_j
            prob_not_trav = 1 - prob_individual[num_trav]
            remaining_revenue += prob_not_trav * base_revenues[num_trav]
            # Then, P(X_j = 1, \pi X_i = 0)*r_j*(1-\lambda)
            others_not = 1 - np.prod([t for num, t in enumerate(prob_individual) if num != num_trav])
            rev_discounted = base_revenues[num_trav] * (1 - _guaranteed_discount)
            remaining_revenue += (prob_individual[num_trav] * others_not) * rev_discounted

        out = [sum(revenue_shared) * probability_shared + remaining_revenue,
               [t[0] for t in discount],
               sum(revenue_shared),
               prob_individual,
               (_rides_row["veh_dist"] * probability_shared +
                sum(_rides_row["individual_distances"]) * (1 - probability_shared)) / 1000]
        max_out = _max_output_func(out)
        out += [max_out]

        if max_out > best[-1]:
            membership_class_probability: list[list] = [[] for _ in range(len(travellers))]
            for num in range(len(travellers)):
                membership_class_probability[num] = [0]*len(_class_membership[0])
                for acceptable_discount in discounts[num][:(discount_indexes[num] + 1)]:
                    membership_class_probability[num][int(acceptable_discount[1])] += 1/_sample_size
            out.insert(-1, membership_class_probability)
            best = out.copy()

    return best


def optimise_discounts(
        rides: pd.DataFrame,
        class_membership: dict,
        times_ns: dict,
        bt_sample: np.array,
        bs_levels: list[float] or dict,
        objective_func: Callable[[list], float] = lambda x: x[0] - x[4],
        min_acceptance: float or None = None,
        guaranteed_discount: float = 0.1,
        fare: float = 0.0015,
        speed: float = 6,
        max_discount: float = 0.5
) -> pd.DataFrame:

    # tqdm.pandas(desc="Accepted discount calculations")

    rides["accepted_discount"] = rides.apply(
        row_acceptable_discount_bayes,
        axis=1,
        _times_non_shared=times_ns,
        _class_membership=class_membership,
        _bs_levels=bs_levels,
        _bt_sample=bt_sample,
        _interval=1,
        _fare=fare
    )

    rides["veh_dist"] = rides["u_veh"] * speed

    # rides["best_profit"] = maximise_profit_bayes_optimised(
    #     _rides=rides,
    #     _class_membership=class_membership,
    #     _sample_size=int(len(bt_sample)/len(class_membership[0].keys())),
    #     _fare=fare,
    #     _guaranteed_discount=guaranteed_discount,
    #     _min_acceptance=min_acceptance
    # )
    #
    # raise Exception('ee')

    # tqdm.pandas(desc="Discount optimisation")
    rides["best_profit"] = rides.apply(row_maximise_profit_bayes,
                                                axis=1,
                                                _class_membership=class_membership,
                                                _sample_size=int(len(bt_sample)/len(class_membership[0].keys())),
                                                _fare=fare,
                                                _max_output_func=objective_func,
                                                _guaranteed_discount=guaranteed_discount,
                                                _min_acceptance=min_acceptance,
                                                _max_discount=max_discount
                                                )

    rides = rides.loc[[t != [0, 0, 0, 0, 0, 0, 0] for t in rides['best_profit']]]
    rides = rides.reset_index(drop=True)

    rides["objective"] = rides['best_profit'].apply(lambda x: x[-1])

    return rides


def bayesian_vot_updated(
        decision: bool,
        pax_id: int,
        apriori_distribution: dict,
        conditional_probs: list,
        distribution_history: dict or False = False
):
    """
    Update dictionary with class membership probabilities according to the decision.
    Calculates for a single traveller.
    :param decision:
    :param pax_id:
    :param apriori_distribution:
    :param conditional_probs:
    :param distribution_history:
    :return: posteriori probabilities
    """
    if decision:
        posteriori_probability = conditional_probs
    else:
        posteriori_probability = [1-t for t in conditional_probs]

    posteriori_probability = [a*b for a, b in
                              zip(posteriori_probability, apriori_distribution[pax_id].values())]
    posteriori_probability = [t/sum(posteriori_probability) for t in posteriori_probability]

    apriori_distribution[pax_id] = {k: v for k, v in
                                    zip(apriori_distribution[pax_id].keys(), posteriori_probability)}

    if distribution_history:
        for _num, key in enumerate(distribution_history['updated'][pax_id].keys()):
            distribution_history['updated'][pax_id][key].append(posteriori_probability[_num])
            distribution_history['all'][pax_id][key].append(posteriori_probability[_num])

    return apriori_distribution


def aggregate_daily_results(
        day_results: dict,
        predicted_satisfaction: dict,
        actual_satisfaction: dict,
        fare: float,
        run_config: dict,
):
    """
    Aggregate results after each day to track system's evolution.
    :param day_results: results from a daily run
    :param predicted_satisfaction: prediction of travellers' satisfaction
    :param actual_satisfaction: tracked, actual travellers satisfaction
    :param fare: per-kilometre price
    :param run_config: configuration for objective calculation
    :return: table with results
    """
    results = pd.Series()
    schedule = day_results['schedules']['objective']
    results['TravellersNo'] = len(day_results['requests'])
    results['Objective'] = sum(schedule['objective'])
    schedule_sharing = schedule.loc[schedule['indexes'].apply(lambda x: len(x) >= 2)]
    schedule_non_sharing = schedule.loc[schedule['indexes'].apply(lambda x: len(x) == 1)]
    schedule_sharing = schedule_sharing.reset_index(inplace=False, drop=True)
    results['SharingTravellerOffer'] = sum(schedule_sharing['indexes'].apply(len))
    results['SharedRidesNo'] = len(schedule_sharing)
    results['ExpectedSharingFraction'] = sum(schedule_sharing['indexes'].apply(len))/len(day_results['requests'])
    results['ExpectedRevenue'] = sum(schedule['best_profit'].apply(lambda x: x[0]))
    results['ExpectedDistance'] = sum(schedule['best_profit'].apply(lambda x: x[4]))
    results['ExpectedSharingRevenue'] = sum(schedule_sharing['best_profit'].apply(lambda x: x[0]))
    results['ExpectedSharingDistance'] = sum(schedule_sharing['best_profit'].apply(lambda x: x[4]))

    actual_rides_no = len(schedule_non_sharing)

    actual_sharing_revenue: float = 0
    actual_sharing_distance: float = 0
    actual_acceptance_rate: float = 0
    actual_rejected_sharing_distance: float = 0
    actual_rejected_sharing_revenue: float = 0
    realised_shared_rides: int = 0
    for _, shared_ride in schedule_sharing.iterrows():
        # if the share ride is realised
        if shared_ride['decision']:
            actual_sharing_revenue += shared_ride['best_profit'][2]
            actual_sharing_distance += shared_ride['veh_dist']/1000
            actual_acceptance_rate += len(shared_ride['indexes'])
            realised_shared_rides += 1
            actual_rides_no += 1
        else:
            actual_rejected_sharing_distance += sum(shared_ride['individual_distances'])/1000
            for pax_no, pax in enumerate(shared_ride['indexes']):
                actual_rides_no += 1
                decision = shared_ride['decisions'][pax_no]
                if decision:
                    actual_acceptance_rate += 1
                    actual_rejected_sharing_revenue += (shared_ride['individual_distances'][pax_no]*
                                             fare*(1-run_config['guaranteed_discount'])/1000)
                else:
                    actual_rejected_sharing_revenue += fare*shared_ride['individual_distances'][pax_no]/1000
    results['ActualSharingRevenue'] = actual_sharing_revenue
    results['ActualRejectedSharingRevenue'] = actual_rejected_sharing_revenue
    results['ActualRevenue'] = (actual_sharing_revenue + actual_rejected_sharing_revenue
                                + (1-run_config['guaranteed_discount'])*fare*sum(schedule_non_sharing['veh_dist'])/1000)
    results['ActualSharingDistance'] = actual_sharing_distance
    results['ActualRejectedSharingDistance'] = actual_rejected_sharing_distance
    results['ActualDistance'] = (actual_sharing_distance + actual_rejected_sharing_distance
                                 + sum(schedule_non_sharing['veh_dist'])/1000)
    results['RealisedSharedRides'] = realised_shared_rides
    results['ActualAcceptanceRate'] = actual_acceptance_rate/results['SharingTravellerOffer']

    results['MeanPredictedSatisfaction'] = np.mean(list(predicted_satisfaction.values()))
    results['MeanActualSatisfaction'] = np.mean(list(actual_satisfaction.values()))
    results['MeanPredictedParticipationProbability'] = np.mean([_sigmoid(t) for t in predicted_satisfaction.values()])
    results['MeanActualParticipationProbability'] = np.mean([_sigmoid(t) for t in actual_satisfaction.values()])

    results['ActualRidesNo'] = actual_rides_no
    results['ActualObjectiveValue'] = results['ActualRevenue']
    results['ActualObjectiveValue'] -= run_config['mileage_sensitivity']*results['ActualDistance']
    results['ActualObjectiveValue'] -= results['ActualRidesNo']*run_config['flat_fleet_cost']

    return results


def maximise_profit_bayes_optimised(
        _rides: pd.DataFrame,
        _class_membership: dict,
        _sample_size: int,
        _fare: float = 0.0015,
        _guaranteed_discount: float = 0.05,
        _min_acceptance: float = 0
):
    """ Function to calculate the expected performance (or its derivatives)
     of a precalculated exmas ride with hardcoded objective though optimised
    --------
    :param _rides: row of exmas rides (potential combination + characteristics)
    :param _class_membership: probability for each agent to belong to a given class
    :param _sample_size: number of samples of value of time for a class
    :param _fare: per-kilometre fare
    :param _guaranteed_discount: when traveller accepts a shared ride and any of the co-travellers
    reject a ride, the traveller is offered
    :param _min_acceptance: minimum acceptance probability
    --------
    :return vector comprising 6 main characteristics when applied discount maximising
    the expected revenue:
    - expected revenue
    - vector of individual discounts
    - revenue from the shared ride if accepted
    - vector of probabilities that individuals accept the shared ride
    - expected distance
    - probability of acceptance when in certain class: [t1 in C1, t1 in C2,...], [t2 in C1, t2 in C2, ...]
    - max output function (by default, profitability) """
    @jit
    def row_calculations(
            _indexes: np.ndarray or list,
            _individual_distances: np.ndarray or list,
            _discounts_values: np.ndarray or list,
            _discounts_probs: np.ndarray or list,
            _veh_dist: float,
            _sample_size: int,
            _fare_km: float,
            _guaranteed_discount: float,
            _min_acceptance: float
    ):
        _travellers_no = len(_indexes)
        if _travellers_no == 1:
            out = [
                _veh_dist*_fare_km*(1-_guaranteed_discount),
                0,
                _veh_dist,
                [1],
                _veh_dist/1000,
                _veh_dist/1000 * (_fare_km-1) * (1 - _guaranteed_discount)
                ]

            return out

        base_revenues = np.array([a*_fare_km for a in _individual_distances])
        best = np.zeros(7)

        for pointer in range(len(_discounts_values)):
            discount_values = _discounts_values[pointer]
            discount_probs = _discounts_probs[pointer]
            effective_price = _fare_km*(1 - discount_values)
            # revenue_shared = [a*b for a,b in zip(
            #     _individual_distances, effective_price
            # )]
            revenue_shared = np.array(_individual_distances) * effective_price
            probability_shared = 1
            for discount_prob in discount_probs:
                probability_shared *= discount_prob
            # probability_shared = np.prod(discount_probs)

            if probability_shared < _min_acceptance:
                continue

            remaining_revenue = 0
            for pax in range(_travellers_no):
                # First, if the P(X_j = 0)*r_j
                prob_no_trav = 1 - discount_probs[pax]
                remaining_revenue += prob_no_trav * base_revenues[pax]
                # Then, P(X_j = 1, \pi X_i = 0)*r_j*(1-\lambda)
                others_not = 1 - np.prod(np.concatenate([discount_probs[:pax], discount_probs[(pax+1):]]))
                rev_discounted = base_revenues[pax] * (1 - _guaranteed_discount)
                remaining_revenue += discount_probs[pax]*others_not*rev_discounted

            out = [
                sum(revenue_shared) * probability_shared + remaining_revenue,
                discount_values,
                sum(revenue_shared),
                discount_probs,
                _veh_dist*probability_shared+sum(_individual_distances)*(1-probability_shared)/1000,
                0
            ]

            out[5] = out[0] - out[4]

            if out[-1] > best[-1]:
                best = out.copy()

        return best


    def _amend_discounts(
            _indexes: list,
            _discounts_single: list,
            _class_membership_prob: dict,
            _sample_size: int
    ):
        if not _discounts_single:
            return [[0, 0]]

        cum = []
        for pax, disc in enumerate(_discounts_single):
            _ = [(t[0], _class_membership_prob[pax][t[1]]/_sample_size) for t in disc]
            cum.append([[d, p] for d, p in zip(
                [t[0] for t in _],
                np.cumsum([t[1] for t in _])
            )])

        out = list(itertools.product(*cum))
        out = [([t[0] for t in o], [t[1] for t in o]) for o in out]

        return out



    if _fare > 1:
        km_fare: float = _fare/1000
    else:
        km_fare: float = _fare

    indexes = _rides['indexes'].to_numpy()
    individual_distances = _rides['individual_distances'].to_numpy()

    amended_discounts = _rides.apply(
        lambda x: _amend_discounts(x['indexes'], x['accepted_discount'],
                                   _class_membership, _sample_size),
        axis=1
    ).to_numpy()

    vehicle_distances = _rides['veh_dist'].to_numpy()

    optimal_discounts = [row_calculations(
        _indexes=a,
        _individual_distances=b,
        _discounts_values=np.array([t[0] for t in c]),
        _discounts_probs=np.array([t[1] for t in c]),
        _veh_dist=d,
        _sample_size=_sample_size,
        _fare_km=km_fare,
        _guaranteed_discount=_guaranteed_discount,
        _min_acceptance=_min_acceptance
    ) for a, b, c, d in zip(indexes, individual_distances, amended_discounts, vehicle_distances)]

    return 0


def check_if_stabilised(
        day_results: dict,
        last_schedule: pd.Series or list,
        stabilised_archive: list
):
    """
    Check if day by day now results present the same schedule
    :param day_results:
    :param last_schedule
    :param stabilised_archive:
    :return:
    """
    _indexes = day_results['schedules']['objective']['indexes']
    if len(_indexes) != len(last_schedule):
        stabilised_archive.append(False)
    else:
        if all(a == b for a, b in zip(_indexes, last_schedule)):
            stabilised_archive.append(True)
        else:
            stabilised_archive.append(False)
    last_schedule = day_results['schedules']['objective']['indexes'].copy()

    return last_schedule, stabilised_archive


def all_class_tracking(
        distribution_history: dict,
        updated_travellers: list,
        all_travellers: list
):
    for traveller in all_travellers:
        if traveller not in updated_travellers:
            for class_id, prob in distribution_history['all'][traveller].items():
                distribution_history['all'][traveller][class_id].append(distribution_history['all'][traveller][class_id][-1])

    return distribution_history


def optimise_discounts_future(
        rides: pd.DataFrame,
        class_membership: dict,
        vot_sample: np.array,
        bs_levels: list[float] or dict,
        travellers_satisfaction: dict,
        objective_func: Callable[[list], float] = lambda x: x[0] - x[4],
        min_acceptance: float or None = None,
        guaranteed_discount: float = 0.1,
        fare: float = 1.5,
        speed: float = 6,
        max_discount: float = 0.5,
        attraction_sensitivity: float = 1
) -> pd.DataFrame:
    rides["accepted_discount"] = rides.apply(
        row_acceptable_discount_bayes,
        axis=1,
        _class_membership=class_membership,
        _bs_levels=bs_levels,
        _vot_sample=vot_sample,
        _speed=speed,
        _fare=fare
    )

    rides["veh_dist"] = rides["u_veh"] * speed

    # tqdm.pandas(desc="Discount optimisation")
    rides["best_profit"] = rides.apply(row_maximise_profit_future,
                                                axis=1,
                                                _class_membership=class_membership,
                                                _sample_vot=vot_sample,
                                                _bs_levels=bs_levels,
                                                _individual_satisfaction=travellers_satisfaction,
                                                _fare=fare,
                                                _speed=speed,
                                                _max_output_func=objective_func,
                                                _guaranteed_discount=guaranteed_discount,
                                                _min_acceptance=min_acceptance,
                                                _max_discount=max_discount,
                                                _attraction_sensitivity=attraction_sensitivity
                                                )

    rides = rides.loc[[t != [0]*9 for t in rides['best_profit']]]
    rides = rides.reset_index(drop=True)

    rides["objective"] = rides['best_profit'].apply(lambda x: x[-1])

    return rides


def row_maximise_profit_future(
        _rides_row: pd.Series,
        _class_membership: dict[dict],
        _max_output_func: Callable[[list], float],
        _sample_vot: list or np.ndarray[int or float],
        _bs_levels: list or np.ndarray[int or float],
        _individual_satisfaction: dict,
        _fare: float = 0.0015,
        _speed: float = 6,
        _probability_single: float = 1,
        _guaranteed_discount: float = 0.05,
        _min_acceptance: float = 0,
        _max_discount: float = 0.5,
        _attraction_sensitivity: float = 1
):
    """
    Function to calculate the expected performance (or its derivatives)
     of a precalculated exmas ride. Apply future value for calculations
    --------
    @param _rides_row: row of exmas rides (potential combination + characteristics)
    @param _class_membership: probability for each agent to belong to a given class
    @param _sample_vot: number of samples of value of time for a class
    @param _bs_levels: list with value of penalty for sharing
    @param _individual_satisfaction: for each traveller probability that one joins the service
    @param _speed: constant speed
    @param _fare: per-kilometre fare
    @param _probability_single: probability that a customer is satisfied with a private
    ride if offered one
    @param _guaranteed_discount: when traveller accepts a shared ride and any of the co-travellers
    reject a ride, the traveller is offered
    @param _max_output_func: specify what is the maximisation objective
    @param _min_acceptance: minimum acceptance probability
    @param _max_discount: maximum allowed sharing discount
    @param _attraction_sensitivity: how much the attraction value is considered in pricing
    --------
    @return vector comprising 6 main characteristics when applied discount maximising
    the expected revenue:
    - expected revenue [0]
    - vector of individual discounts [1]
    - revenue from the shared ride if accepted [2]
    - vector of probabilities that individuals accept the shared ride [3]
    - expected distance [4]
    - attraction value [5]
    - expected vehicle quantity [6]
    - probability of acceptance when in certain class: [t1 in C1, t1 in C2,...], [t2 in C1, t2 in C2, ...] [7]
    - maximal vot levels to accept [8]
    - max output function value [9]
    """
    assert 0.1 < _fare < 10, ValueError('The fare should be per kilometre,'
                                        'assumed to be in range (0.1, 10)')
    _fare_meter: float = _fare/1000

    no_travellers = len(_rides_row["indexes"])
    if no_travellers == 1:
        out = [_probability_single * _rides_row["veh_dist"] * _fare_meter * (1 - _guaranteed_discount),
               _guaranteed_discount,
               _rides_row["veh_dist"] * _fare_meter * (1 - _guaranteed_discount),
               [_probability_single],
               _rides_row["veh_dist"] / 1000 * _probability_single,
               0,
               1,
               [1]*len(_rides_row["indexes"]),
               [100]
               ]
        out += [_max_output_func(out)]
        return out

    else:
        non_shared_values = []
        for pax_no in range(no_travellers):
            out = [_probability_single * _rides_row["individual_distances"][pax_no]
                   * _fare_meter * (1 - _guaranteed_discount),
                   _guaranteed_discount,
                   _rides_row["individual_distances"][pax_no] * _fare_meter * (1 - _guaranteed_discount),
                   [_probability_single],
                   _rides_row["individual_distances"][pax_no] / 1000 * _probability_single,
                   0,
                   1,
                   [1]*len(_rides_row["indexes"]),
                   [100]
                 ]
            non_shared_values.append(_max_output_func(out))

    # Extract features
    travellers = _rides_row["indexes"]
    base_revenues = {num: _rides_row["individual_distances"][num] * _fare_meter for num, pax in
                     enumerate(_rides_row['indexes'])}
    _sample_size = int(len(_sample_vot) / len(_class_membership[0].keys()))
    avg_vot = {num: sum(_class_membership[pax][vot_class]*vot for vot, vot_class in _sample_vot)/_sample_size
               for num, pax in enumerate(_rides_row['indexes'])}
    expected_time_utilities = [avg_vot[num]/3600*(_bs_levels[len(travellers)]*
                                                     _rides_row['individual_times'][num] -
                                                     _rides_row['individual_distances'][num]/_speed)
                                  for num in range(len(travellers))]
    travellers_satisfaction=[_individual_satisfaction[sorted(_individual_satisfaction.keys())[-1]][t]
                             for t in travellers]

    # Prepare discount space
    discounts = _rides_row["accepted_discount"].copy()
    discounts = [[(t[0], t[1], t[2]) if t[0] > _guaranteed_discount else (_guaranteed_discount, t[1], t[2])
                 for t in d] for d in discounts] # at least guaranteed discount
    discounts_indexes = list(itertools.product(*[range(len(t)) for t in discounts]))

    # Variable for tracking
    best = [0]*9

    discount_indexes: tuple[int]
    for discount_indexes in discounts_indexes:
        # Start with the effectively shared ride
        discount = [discounts[_num][_t] for _num, _t in enumerate(discount_indexes)]
        if any(t[0] > _max_discount for t in discount):
            continue
        eff_price = [_fare_meter * (1 - t[0]) for t in discount]
        revenue_shared = [a * b for a, b in
                          zip(_rides_row["individual_distances"], eff_price)]
        probability_shared = 1
        prob_individual = [.0] * len(discount)

        for num, pax in enumerate(travellers):
            prob_individual[num] = (sum(_class_membership[pax][t[1]]
                                  for t in discounts[num][:(discount_indexes[num]+1)])/
                               _sample_size)
            probability_shared *= prob_individual[num]

        if probability_shared < _min_acceptance:
            continue

        remaining_revenue = 0
        for num_trav in range(len(discount)):
            # First, if the P(X_j = 0)*r_j
            prob_not_trav = 1 - prob_individual[num_trav]
            remaining_revenue += prob_not_trav * base_revenues[num_trav]
            # Then, P(X_j = 1, \pi X_i = 0)*r_j*(1-\lambda)
            others_not = 1 - np.prod([t for num, t in enumerate(prob_individual) if num != num_trav])
            rev_discounted = base_revenues[num_trav] * (1 - _guaranteed_discount)
            remaining_revenue += (prob_individual[num_trav] * others_not) * rev_discounted

        out = [sum(revenue_shared) * probability_shared + remaining_revenue,
               [t[0] for t in discount],
               sum(revenue_shared),
               prob_individual,
               (_rides_row["veh_dist"] * probability_shared +
                sum(_rides_row["individual_distances"]) * (1 - probability_shared)) / 1000,
               0,
               probability_shared + (1-probability_shared)*no_travellers]

        max_out = _max_output_func(out)

        # Attraction value. First, shared option
        monetary_savings = [disc*dist*_fare_meter for disc, dist in
                            zip(out[1], _rides_row["individual_distances"])]
        delta_utilities = [a - b for a, b in zip(monetary_savings, expected_time_utilities)]
        probability_potential = [_sigmoid(new + cur)
                                for new, cur in zip(delta_utilities, travellers_satisfaction)]
        probability_increase = [c - _sigmoid(d) for c, d in zip(probability_potential,travellers_satisfaction)]
        probability_not_others = [probability_increase[j] *
                                  (1 - np.prod(probability_potential[:j] + probability_potential[(j+1):]))
                                  for j in range(len(probability_potential))]
        attraction_value = ((np.prod(probability_potential) -
                            np.prod([_sigmoid(s) for s in travellers_satisfaction])) *
                            max_out) # shared ride
        attraction_value += sum(a*b for a, b in zip(probability_not_others, non_shared_values)) # single rides

        out[5] = [attraction_value]
        out += [max_out + attraction_value]

        if max_out > best[-1]:
            membership_class_probability: list[list] = [[] for _ in range(len(travellers))]
            for num in range(len(travellers)):
                membership_class_probability[num] = [0]*len(_class_membership[0])
                for acceptable_discount in discounts[num][:(discount_indexes[num] + 1)]:
                    membership_class_probability[num][int(acceptable_discount[1])] += 1/_sample_size
            out.insert(-1, membership_class_probability)
            out.insert(-1, [t[2] for t in discount])
            best = out.copy()

    return best


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def update_satisfaction(
        predicted_travellers_satisfaction_day: dict,
        actual_travellers_satisfaction_day: dict,
        schedule_row: pd.Series,
        predicted_class_distribution: dict,
        predicted_satisfaction: dict,
        actual_satisfaction: dict,
        vot_sample: dict,
        bs_levels: list,
        speed: float = 6,
        fare: float = 1.5
):
    assert 0.1 < fare < 10, ValueError('The fare should be per kilometre,'
                                        'assumed to be in range (0.1, 10)')
    _fare_meter: float = fare / 1000

    _sample_size = int(len(vot_sample) / len(predicted_class_distribution[0].keys()))

    delta_perceived_times = [bs_levels[len(schedule_row['indexes'])] * schedule_row['individual_times'][num] -
                             schedule_row['individual_distances'][num] / speed
                             for num in range(len(schedule_row['indexes']))]

    monetary_savings = [disc * dist * _fare_meter for disc, dist in
                        zip(schedule_row['best_profit'][1], schedule_row["individual_distances"])]

    # First, the estimated satisfaction
    max_accepted_vot = schedule_row['best_profit'][8]
    conditions_vot = [vot_sample[:,0] <= vot if decision
                      else vot_sample[:,0] > vot for decision, vot
                      in zip(schedule_row['decisions'], max_accepted_vot)]
    feasible_vot = [vot_sample[conditions_vot[num]] for num in range(len(max_accepted_vot))]
    avg_vot = [sum(predicted_class_distribution[pax][vot_class]* vot
                        for vot, vot_class in feasible_vot[num]) /
               sum(predicted_class_distribution[pax][vot_class]
                   if predicted_class_distribution[pax][vot_class]>0
                   else 0
                   for vot, vot_class in feasible_vot[num])
               for num, pax in enumerate(schedule_row['indexes'])]
    expected_time_utilities = [avg_vot[num] / 3600 * delta_perceived_times[num]
                               for num in range(len(schedule_row['indexes']))]
    delta_utilities = [a - b for a, b in zip(monetary_savings, expected_time_utilities)]

    for num, pax in enumerate(schedule_row['indexes']):
        if all(schedule_row['best_profit'][8]) or (not schedule_row['best_profit'][8][num]):
            predicted_travellers_satisfaction_day[pax] = delta_utilities[num] + predicted_satisfaction[pax]

    # Second, the actual satisfaction
    for num, pax in enumerate(schedule_row['indexes']):
        expected_time_utility = schedule_row['sampled_vot'][num] / 3600 * delta_perceived_times[num]
        if all(schedule_row['best_profit'][8]) or (not schedule_row['best_profit'][8][num]):
            actual_travellers_satisfaction_day[pax] = (
                    monetary_savings[num] - expected_time_utility + actual_satisfaction[pax])

    return predicted_travellers_satisfaction_day, actual_travellers_satisfaction_day


def reliability_performance_analysis(
        schedules: list,
        run_config: dict,
        exmas_config: dict
):
    """
    Check the stability of results if travellers choose differently (accept/reject)
    :param schedules:
    :param run_config
    :param exmas_config
    :return:
    """
    def _extract_scenarios(_ride_row):
        def _func(_d, _p):
            _prob = 1
            for _dd, _pp in zip(_d, _p):
                if _dd == 1:
                    _prob *= _pp
                else:
                    _prob *= (1 - _pp)
            return _d, _prob

        decisions = list(itertools.product(range(2), repeat=len(_ride_row['indexes'])))
        scenarios = []
        for decision_vect in decisions:
            scenarios.append(_func(decision_vect, _ride_row['best_profit'][3]))

        return scenarios


    def _calculate_mean_and_variance(realisations):
        EX = sum(p * x for p, x in realisations)
        EX2 = sum(p * np.power(x, 2) for p, x in realisations)
        EX_2 = np.power(sum(p * x for p, x in realisations), 2)
        return EX, EX2 - EX_2

    if exmas_config['price'] > 1:
        fare_metres: float = exmas_config['price']/1000
    else:
        fare_metres: float = exmas_config['price']

    daily_mean = []
    daily_variance = []

    for schedule in schedules:
        day_mean = sum(schedule.loc[[len(t) == 1 for t in schedule['indexes']], 'objective'])
        day_variance = 0

        schedule_sh = schedule.loc[[len(t) > 1 for t in schedule['indexes']]]

        for num, ride in schedule_sh.iterrows():
            ride_realisations = []
            decision_scenarios = _extract_scenarios(ride)
            for decision in decision_scenarios:
                if all(decision[0]):
                    ride_realisations += [(decision[1],
                                          ride['best_profit'][2] -
                                          run_config['mileage_sensitivity']*ride['veh_dist']/1000 -
                                          run_config['flat_fleet_cost'])]
                else:
                    _revenues = [fare_metres * (1 - run_config['guaranteed_discount']) * dist
                              if ind_decision else fare_metres * dist
                              for dist, ind_decision in zip(ride['individual_distances'], decision[0])]
                    ride_realisations += [(decision[1],
                                          sum(_revenues) -
                                          run_config['mileage_sensitivity']*sum(ride['individual_distances'])/1000 -
                                          run_config['flat_fleet_cost']*len(decision[0]))]
            mean, var = _calculate_mean_and_variance(ride_realisations)

            day_mean += mean
            day_variance += var

        daily_mean.append(day_mean)
        daily_variance.append(day_variance)

    return daily_mean, daily_variance


def post_run_analysis(
        class_membership_stability: dict,
        actual_class_membership: dict,
        results_daily: pd.DataFrame,
        all_results_aggregated: dict,
        predicted_travellers_satisfaction:dict,
        actual_travellers_satisfaction:dict,
        run_config: dict,
        exmas_params: dict,
        out_path: str,
        args: dict or argparse.Namespace,
        **kwargs
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    x_ticks = kwargs.get('x_ticks', [round(t) for t in np.arange(0, 22, 2)])
    x_ticks_labels = kwargs.get('x_ticks_labels', [str(round(t) + 1) for t in np.arange(0, 22, 2)])

    # KPIs to latex
    _results_daily = results_daily.copy()
    if 'metric' in _results_daily.columns:
        _results_daily = _results_daily.set_index('metric')
    columns_to_int = ['TravellersNo', 'SharingTravellerOffer', 'SharedRidesNo', 'RealisedSharedRides', 'ActualRidesNo']
    for column in _results_daily.columns:
        if column in columns_to_int:
            _results_daily[column] = (_results_daily[column].apply(round)).apply(int)
        else:
            _results_daily[column] = _results_daily[column].apply(round, ndigits=2)

    _results_daily.to_latex(out_path + 'results_daily.txt', float_format="%.2f")

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
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig(out_path + 'class_error_iteration.' + args.plot_format, dpi=args.plot_dpi)
    plt.close()

    heatmap_df = pd.DataFrame()
    for day_no, day_err in enumerate(error_by_day):
        heatmap_df[day_no] = _count_for_heatmap(day_err, [round(t, 1) for t in np.arange(0, 1.1, 0.1)])

    sns.heatmap(heatmap_df, cmap=LinearSegmentedColormap.from_list('', ['white', 'darkorange']))
    plt.yticks(np.arange(0, 11, 1), labels=[str(round(100*t)) + '%' for t in np.arange(0, 1.1, 0.1)])
    plt.tight_layout()
    plt.savefig(out_path + 'class_error_heatmap_iteration.' + args.plot_format, dpi=args.plot_dpi)
    plt.close()

    # Heatmap by day
    error_by_day = [[1 - class_membership_stability['all'][pax][actual_class_membership[pax]][day]
                       for pax in actual_class_membership.keys()]
               for day in range(len(class_membership_stability['all'][0][0]))]
    heatmap_df = pd.DataFrame()
    for day_no, day_err in enumerate(error_by_day):
        heatmap_df[day_no] = _count_for_heatmap(day_err, [round(t, 1) for t in np.arange(0, 1.1, 0.1)])
    sns.heatmap(heatmap_df, cmap=LinearSegmentedColormap.from_list('', ['white', 'darkorange']),
                cbar_kws={'ticks': np.arange(0, 280, 40)})
    plt.yticks(np.arange(0, 11, 1), labels=[str(round(100*t)) + '%' for t in np.arange(0, 1.1, 0.1)])
    plt.tight_layout()
    plt.savefig(out_path + 'class_error_heatmap_day.' + args.plot_format, dpi=args.plot_dpi)
    plt.close()

    try:
        results_daily = results_daily.rename(columns={'metric': 'day'}).set_index('day').T
    except KeyError:
        results_daily = results_daily.T

    # Actual profit
    plt.plot(results_daily['ActualObjectiveValue'].to_list(), label='Actual Profit')

    # Reliability of performance prediction
    objective_mean, objective_variance = reliability_performance_analysis(
        schedules=[t['schedules']['objective'] for t in all_results_aggregated],
        run_config=run_config,
        exmas_config=exmas_params
    )

    plt.errorbar(x=range(len(objective_mean)),
                 y=objective_mean,
                 yerr=[t / 2 for t in objective_variance],
                 label='Expected Profit')
    plt.xticks(x_ticks,
               labels=x_ticks_labels)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path + 'service_performance.' + args.plot_format, dpi=args.plot_dpi)
    plt.close()

    # Satisfaction
    fig, ax1 = plt.subplots()
    y11 = [0] + results_daily['MeanActualSatisfaction'].tolist()
    y12 = [0] + results_daily['MeanPredictedSatisfaction'].tolist()
    y21 = [0.5] + results_daily['MeanActualParticipationProbability'].tolist()
    y22 = [0.5] + results_daily['MeanPredictedParticipationProbability'].tolist()

    ax1.set_xlabel('Day')
    ax1.set_ylabel('Satisfaction', color='black')
    ax1.plot(range(0, run_config['no_days']+1, 1), y11, color='black', ls=':')
    ax1.plot(range(0, run_config['no_days']+1, 1), y12, color='black', ls='--', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='black')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Participation Probability', color='darkorange')
    ax2.plot(range(0, run_config['no_days']+1, 1), y21, color='darkorange', ls=':')
    ax2.plot(range(0, run_config['no_days']+1, 1), y22, color='darkorange', ls='--', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, 1)
    plt.legend([('darkorange', ':'), ('darkorange', '--')], ['Actual', 'Predicted'],
               handler_map={tuple: AnyObjectHandler()}, loc='upper left')
    plt.xticks(np.arange(0, 25, 5), [str(round(t)) for t in np.arange(0, 25, 5)])
    plt.tight_layout()
    plt.savefig(out_path + 'satisfaction.' + args.plot_format, dpi=args.plot_dpi)
    plt.close()

    # Discounts per day
    fitted_discounts_sel = [[b[1] for b in t['schedules']['objective']['best_profit']] for t in all_results_aggregated]
    fitted_discounts_all = [[b[1] for b in t['rides']['best_profit']] for t in all_results_aggregated]
    fitted_sel_discounts = _extract_discounts(fitted_discounts_sel)
    fitted_all_discounts = _extract_discounts(fitted_discounts_all)
    thresholds = np.arange(0, 0.45, 0.05)

    for fi_dat, fi_name in zip([fitted_sel_discounts, fitted_all_discounts], ['selected', 'all']):
        heatmap_df = pd.DataFrame()
        for day_no, day_err in enumerate(fi_dat):
            heatmap_df[day_no] = _count_for_heatmap(day_err, thresholds)

        sns.heatmap(heatmap_df, cmap=LinearSegmentedColormap.from_list('', ['white', 'darkorange']))
        plt.yticks(np.arange(0, len(thresholds), 1), labels=[str(round(100*t)) + '%' for t in thresholds])
        plt.tight_layout()
        plt.savefig(out_path + 'discounts_' + fi_name + '.' + args.plot_format, dpi=args.plot_dpi)
        plt.close()

    # Daily actual sharing
    y1 = [round(a/b, 3) for a, b in zip(results_daily['SharingTravellerOffer'], results_daily['TravellersNo'])]
    y2 = [sum(all(d) for d in x['decisions'])/len(x) for x in
          [t['schedules']['objective'].loc[[len(r)>1 for r in t['schedules']['objective']['indexes']]]
           for t in all_results_aggregated]]
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Travellers offered shared rides', color='black')
    ax1.plot(range(len(y1)), y1, color='black', lw=1)
    # ax1.scatter(range(len(y1)), y1, color='black')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accepted shared rides (fraction)', color='darkorange')
    ax2.plot(range(len(y1)), y2, color='darkorange', lw=1)
    # ax2.scatter(range(len(y1)), y2, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    plt.xticks(x_ticks,
               labels=x_ticks_labels)
    plt.tight_layout()
    plt.savefig(out_path + 'sharing_acceptance.' + args.plot_format, dpi=args.plot_dpi)
    plt.close()



def _count_for_heatmap(
        vector: list or np.ndarray or pd.Series or tuple,
        thresholds: list or tuple,
):
    out = [0]*(len(thresholds)-1)
    _vector = sorted(vector)
    for j in range(len(out)):
        out[j] = bisect_right(_vector, thresholds[j+1]) - bisect_left(_vector, thresholds[j])

    return out


def _extract_discounts(
        stacked_list: list,
        with_single: bool = False
):
    fitted_discounts = []
    for day_discount_list in stacked_list:
        day_discounts = []
        for discount in day_discount_list:
            if isinstance(discount, list):
                day_discounts += discount
            else:
                if with_single:
                    day_discounts += [discount]
        fitted_discounts.append(day_discounts.copy())
    return fitted_discounts

class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                           linestyle=orig_handle[1], color='k')
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height],
                           color=orig_handle[0], linestyle=orig_handle[1])
        return [l1, l2]



def benchmarks(
        all_results_aggregated: list,
        _results_daily: pd.DataFrame,
        _actual_satisfaction: dict,
        _actual_classes: dict,
        _run_config: dict,
        _flat_discount: float = 0.2
) -> pd.DataFrame():
    _fare = _run_config['price']
    _actual_satisfaction = {int(k): v for k,v in _actual_satisfaction.items()}

    if 'metric' in _results_daily.columns:
        _results_daily = _results_daily.set_index('metric')

    last_day = len(all_results_aggregated)-1

    output_dict = {}
    total_distance0 = sum(all_results_aggregated[0]['requests']['dist'])/1000
    total_distance_l = sum(all_results_aggregated[-1]['requests']['dist'])/1000

    output_dict['hetero'] = pd.Series({
        'ExpectedProfit': _results_daily.loc['Objective'][0],
        'Profit': _results_daily.loc['ActualObjectiveValue'][0],
        'Occupancy': _results_daily.loc['TravellersNo'][0]/_results_daily.loc['ActualRidesNo'][0],
        'DistanceSaved': total_distance0 - _results_daily.loc['ActualDistance'][0],
        'AcceptanceRate': _results_daily.loc['ActualAcceptanceRate'][0],
        'MeanParticipationProbabilityChange':
            np.mean([_sigmoid(t) - 0.5 for t in _actual_satisfaction[1].values() if t!=0])
    })

    output_dict['full'] = pd.Series({
        'ExpectedProfit': _results_daily.loc['Objective'][last_day],
        'Profit': _results_daily.loc['ActualObjectiveValue'][last_day],
        'Occupancy': _results_daily.loc['TravellersNo'][last_day]/_results_daily.loc['ActualRidesNo'][last_day],
        'DistanceSaved': total_distance_l - _results_daily.loc['ActualDistance'][last_day],
        'AcceptanceRate': _results_daily.loc['ActualAcceptanceRate'][last_day],
        'MeanParticipationProbabilityChange':
            np.mean([_sigmoid(t) - 0.5 for t in _actual_satisfaction[last_day].values() if t != 0])
    })

    from ExMAS.probabilistic_exmas import match
    rides0 = all_results_aggregated[0]['rides'].copy()

    def acc_flat_prob(r_row, _config):
        if len(r_row['indexes']) == 1:
            return 1

        prob_individual = [.0] * len(r_row['indexes'])

        for num, pax in enumerate(r_row['indexes']):
            acc_disc_place = bisect_right([t[0] for t in r_row['accepted_discount'][num]], _flat_discount)
            prob_individual[num] = (sum(_config['class_probs'][int(t[1])]
                                        for t in r_row['accepted_discount'][num][:(acc_disc_place + 1)])
                                    /_config['sample_size'])

        return prob_individual

    rides0['acc_prob'] = rides0.apply(acc_flat_prob, _config=_run_config, axis=1)

    def exmas_obj(r_row):
        if len(r_row['indexes']) == 1:
            return (((1-_run_config['guaranteed_discount'])*_fare - _run_config['mileage_sensitivity'])
                    *r_row['veh_dist']/1000 - _run_config['flat_fleet_cost'])

        ex_rev = np.prod(r_row['acc_prob'])
        ex_rev *= _fare*(1-_flat_discount)*sum(r_row['individual_distances'])/1000
        ex_cost = np.prod(r_row['acc_prob']) * (_run_config['mileage_sensitivity'] * r_row['veh_dist']/1000 +
                                                _run_config['flat_fleet_cost'])
        ex_cost += (1 - np.prod(r_row['acc_prob'])) * (_run_config['mileage_sensitivity'] *
                                                       sum(r_row['individual_distances'])/1000)
        ex_cost += (1 - np.prod(r_row['acc_prob'])) * len(r_row['indexes']) * _run_config['flat_fleet_cost']

        for num_trav in range(len(r_row['indexes'])):
            prob_not_trav = 1 - r_row['acc_prob'][num_trav]
            ex_rev += prob_not_trav *_fare*r_row['individual_distances'][num_trav]/1000
            # Then, P(X_j = 1, \pi X_i = 0)*r_j*(1-\lambda)
            others_not = 1 - np.prod([t for num, t in enumerate(r_row['acc_prob']) if num != num_trav])
            rev_discounted = _fare*(1-_run_config['guaranteed_discount'])*r_row['individual_distances'][num_trav]/1000
            ex_rev += (r_row['acc_prob'][num_trav] * others_not) * rev_discounted

        return ex_rev - ex_cost

    rides0['exmas_obj'] = rides0.apply(exmas_obj, axis=1)

    requests_copy = all_results_aggregated[0]['requests'].copy()

    schedule_exmas = matching_function_light(
        rides0, requests_copy, 'exmas_obj', 'max'
    )

    folder = _run_config['path_results'] + 'Step_0/'
    sampled_vot = all_results_aggregated[0]['schedules']['objective'].apply(
        lambda x: [(t, k) for t, k in zip(x['indexes'], x['sampled_vot'])], axis=1)
    sampled_vot = [a for b in sampled_vot for a in b]
    sampled_vot = {k: v for k, v in sampled_vot}

    schedule_exmas['sampled_vot'] = schedule_exmas['indexes'].apply(
        lambda x: [sampled_vot[t] for t in x]
    )

    schedule_exmas['decisions'] = schedule_exmas['sampled_vot'].apply(
        lambda x: [t <= _flat_discount if len(x)>1 else True for t in x ]
    )
    schedule_exmas['decision'] = schedule_exmas['decisions'].apply(all)

    def actual_utility_flat(r_row, _flat_disc, _config, _fare):
        if len(r_row['indexes']) == 1:
            return [0]
        pfs = _config['pfs_levels'][len(r_row['indexes'])]
        out = [0]*len(r_row['indexes'])
        for n_pax, pax in enumerate(r_row['indexes']):
            out[n_pax] = _flat_disc*r_row['individual_distances'][n_pax]/1000*_fare
            out[n_pax] -= r_row['sampled_vot'][n_pax]/3600*(
                r_row['individual_times'][n_pax]*pfs -
                r_row['individual_distances'][n_pax]/_run_config['avg_speed'])

        return out

    schedule_exmas['actual_utility'] = schedule_exmas.apply(
        actual_utility_flat, _config=_run_config, _flat_disc=_flat_discount,
        _fare=_fare, axis=1)

    def actual_profit(r_row):
        if len(r_row['indexes']) == 1:
            return r_row['exmas_obj']

        if r_row['decision']:
            out = sum(r_row['individual_distances'])*_fare*(1-_flat_discount)/1000
            out -= _run_config['mileage_sensitivity']*sum(r_row['individual_distances'])/1000
            out -= _run_config['flat_fleet_cost']
        else:
            out = 0
            for num, decision in enumerate(r_row['decisions']):
                if decision:
                    out += _fare*(1-_run_config['guaranteed_discount'])*r_row['individual_distances'][num]/1000
                    out -= _run_config['mileage_sensitivity']*r_row['individual_distances'][num]/1000
                    out -= _run_config['flat_fleet_cost']
                else:
                    out += _fare*r_row['individual_distances'][num]/1000
                    out -= _run_config['mileage_sensitivity']*r_row['individual_distances'][num]/1000
                    out -= _run_config['flat_fleet_cost']

        return out

    schedule_exmas['actual_profit'] = schedule_exmas.apply(actual_profit, axis=1)

    schedule_exmas['no_rides'] = schedule_exmas['decisions'].apply(lambda x: 1 if all(x) else len(x))

    schedule_exmas['distance_saved'] = schedule_exmas.apply(
        lambda x: (sum(x['individual_distances']) - x['veh_dist'])/1000 if x['decision'] else 0,
        axis=1
    )

    schedule_exmas_sh = schedule_exmas.loc[[len(t) > 1 for t in schedule_exmas['indexes']]]

    output_dict['exmas'] = pd.Series({
        'ExpectedProfit': sum(schedule_exmas['exmas_obj']),
        'Profit': sum(schedule_exmas['actual_profit']),
        'Occupancy': _results_daily.loc['TravellersNo'][0]/sum(schedule_exmas['no_rides']),
        'DistanceSaved': sum(schedule_exmas['distance_saved']),
        'AcceptanceRate': len(schedule_exmas_sh.loc[schedule_exmas_sh['decision']])/len(schedule_exmas_sh),
        'MeanParticipationProbabilityChange':
            np.mean([_sigmoid(t) - 0.5
                     for t in [a for b in schedule_exmas['actual_utility'] for a in b]
                     if t != 0])
    })

    print(output_dict)


