""" Function to support the dynamic pricing algorithm """
import secrets
from typing import Callable, Tuple, Any
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit

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
        _times_non_shared: dict,
        _class_membership: dict,
        _bs_levels: dict or list,
        _bt_sample: list[list] or list[tuple],
        _interval: int,
        _fare: float
) -> list:
    """
    Samples discounts for which clients accept rides
    at different discount levels
    @param _rides_row: the function works on rows of pd.Dataframe
    @param _times_non_shared: dictionary with individual rides
    obtained as dict(requests["ttrav"])
    @param _class_membership: dictionary with membership probability
    for each traveller for each behavioural class
    @param _bs_levels: penalty for sharing: subject to the number of co-travellers
    @param _bt_sample: sampled value of time across the population
    @param _interval: once per how many items from the list
    should be returned for the final sample
    @param _fare: price per metre
    @return: acceptable discount levels (progressive with probability)
    """
    no_travellers = len(_rides_row["indexes"])
    if no_travellers == 1:
        return []

    out = []

    for no, trav in enumerate(_rides_row["indexes"]):
        pax_out = _rides_row["individual_times"][no]
        pax_out *= _bs_levels[len(_rides_row["indexes"])]
        pax_out -= _times_non_shared[trav]
        pax_out = [(t[0] * pax_out, t[1]) for t in _bt_sample]
        pax_out = [(t[0], t[1]) if t[0] >= 0 else (0, t[1]) for t in pax_out]
        pax_out = [(t[0]/(_fare * _rides_row["individual_distances"][no]), t[1]) for t in pax_out]
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
        _kmFare: float = _fare/1000
    else:
        _kmFare: float = _fare

    no_travellers = len(_rides_row["indexes"])
    if no_travellers == 1:
        out = [_probability_single * _rides_row["veh_dist"] * _kmFare * (1 - _guaranteed_discount),
               0,
               _rides_row["veh_dist"] * _kmFare * (1 - _guaranteed_discount),
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
    base_revenues = {num: _rides_row["individual_distances"][num] * _kmFare for num, t in
                     enumerate(_rides_row['indexes'])}
    best = [0, 0, 0, 0, 0, 0, 0]

    discount_indexes: tuple[int]
    for discount_indexes in discounts_indexes:
        # Start with the effectively shared ride
        discount = [discounts[_num][_t] for _num, _t in enumerate(discount_indexes)]
        if any(t[0] > _max_discount for t in discount):
            continue
        eff_price = [_kmFare * (1 - t[0]) for t in discount]
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

    posteriori_probability = [a*b for a, b in zip(posteriori_probability, apriori_distribution[pax_id].values())]
    posteriori_probability = [t/sum(posteriori_probability) for t in posteriori_probability]

    apriori_distribution[pax_id] = {k: v for k, v in
                                    zip(apriori_distribution[pax_id].keys(), posteriori_probability)}

    if distribution_history:
        for _num, key in enumerate(distribution_history['updated'][pax_id].keys()):
            distribution_history['updated'][pax_id][key].append(posteriori_probability[_num])

    return apriori_distribution


def aggregate_daily_results(
        day_results: dict,
        decisions: list,
        fare: float,
        guaranteed_discount: float = 0.05
):
    """
    Aggregate results after each day to track system's evolution.
    :param day_results: results from a daily run
    :param decisions: list of decisions: whether a shared ride is realised or not
    :param fare: per-kilometre price
    :param guaranteed_discount: discount for traveller who accepted sharing mode
    and their co-traveller rejected
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


    actual_sharing_revenue: float = 0
    actual_sharing_distance: float = 0
    actual_acceptance_rate: float = 0
    actual_rejected_sharing_distance: float = 0
    actual_rejected_sharing_revenue: float = 0
    realised_shared_rides: int = 0
    for decision_indicator, shared_ride in schedule_sharing.iterrows():
        # if the share ride is realised
        if all(decisions[decision_indicator]):
            actual_sharing_revenue += shared_ride['best_profit'][2]
            actual_sharing_distance += shared_ride['veh_dist']/1000
            actual_acceptance_rate += len(shared_ride['indexes'])
            realised_shared_rides += 1
        else:
            actual_rejected_sharing_distance += sum(shared_ride['individual_distances'])/1000
            for pax_no, pax in enumerate(shared_ride['indexes']):
                decision = decisions[decision_indicator][pax_no]
                if decision:
                    actual_acceptance_rate += 1
                    actual_rejected_sharing_revenue += (shared_ride['individual_distances'][pax_no]*
                                             fare*(1-guaranteed_discount)/1000)
                else:
                    actual_rejected_sharing_revenue += fare*shared_ride['individual_distances'][pax_no]/1000
    results['ActualSharingRevenue'] = actual_sharing_revenue
    results['ActualRejectedSharingRevenue'] = actual_rejected_sharing_revenue
    results['ActualRevenue'] = (actual_sharing_revenue + actual_rejected_sharing_revenue
                                + (1-guaranteed_discount)*fare*sum(schedule_non_sharing['veh_dist'])/1000)
    results['ActualSharingDistance'] = actual_sharing_distance
    results['ActualRejectedSharingDistance'] = actual_rejected_sharing_distance
    results['ActualDistance'] = (actual_sharing_distance + actual_rejected_sharing_distance
                                 + sum(schedule_non_sharing['veh_dist'])/1000)
    results['RealisedSharedRides'] = realised_shared_rides
    results['ActualAcceptanceRate'] = actual_acceptance_rate/results['SharingTravellerOffer']

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
        times_ns: dict,
        bt_sample: np.array,
        bs_levels: list[float] or dict,
        travellers_satisfaction: dict,
        ns_utilities: dict,
        objective_func: Callable[[list], float] = lambda x: x[0] - x[4],
        min_acceptance: float or None = None,
        guaranteed_discount: float = 0.1,
        fare: float = 0.0015,
        speed: float = 6,
        max_discount: float = 0.5
) -> pd.DataFrame:
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

    # tqdm.pandas(desc="Discount optimisation")
    rides["best_profit"] = rides.apply(row_maximise_profit_future,
                                                axis=1,
                                                _class_membership=class_membership,
                                                _sample_vot=bt_sample,
                                                _bs_levels=bs_levels,
                                                _individual_satisfaction=travellers_satisfaction,
                                                _ns_utilities=ns_utilities,
                                                _fare=fare,
                                                _speed=speed,
                                                _max_output_func=objective_func,
                                                _guaranteed_discount=guaranteed_discount,
                                                _min_acceptance=min_acceptance,
                                                _max_discount=max_discount
                                                )

    rides = rides.loc[[t != [0]*8 for t in rides['best_profit']]]
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
        _ns_utilities: dict,
        _fare: float = 0.0015,
        _speed: float = 6,
        _probability_single: float = 1,
        _guaranteed_discount: float = 0.05,
        _min_acceptance: float = 0,
        _max_discount: float = 0.5
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
    @param _ns_utilities: utility of non-shared rides per traveller
    @param _guaranteed_discount: when traveller accepts a shared ride and any of the co-travellers
    reject a ride, the traveller is offered
    @param _max_output_func: specify what is the maximisation objective
    @param _min_acceptance: minimum acceptance probability
    @param _max_discount: maximum allowed sharing discount
    --------
    @return vector comprising 6 main characteristics when applied discount maximising
    the expected revenue:
    - expected revenue [0]
    - vector of individual discounts [1]
    - revenue from the shared ride if accepted [2]
    - vector of probabilities that individuals accept the shared ride [3]
    - expected distance [4]
    - future value [5]
    - probability of acceptance when in certain class: [t1 in C1, t1 in C2,...], [t2 in C1, t2 in C2, ...] [6]
    - max output function (by default, profitability) [7]
    """
    if _fare > 1:
        _kmFare: float = _fare/1000
    else:
        _kmFare: float = _fare

    no_travellers = len(_rides_row["indexes"])
    if no_travellers == 1:
        out = [_probability_single * _rides_row["veh_dist"] * _kmFare * (1 - _guaranteed_discount),
               0,
               _rides_row["veh_dist"] * _kmFare * (1 - _guaranteed_discount),
               [_probability_single],
               _rides_row["veh_dist"] / 1000 * _probability_single,
               [1]*len(_class_membership[0])
               ]
        out += [_max_output_func(out)]
        return out

    else:
        non_shared_values = []
        for pax_no in range(no_travellers):
            out = [_probability_single * _rides_row["individual_distances"][pax_no]
                 * _kmFare * (1 - _guaranteed_discount),
                 0,
                 _rides_row["individual_distances"][pax_no] * _kmFare * (1 - _guaranteed_discount),
                 [_probability_single],
                 _rides_row["individual_distances"][pax_no] / 1000 * _probability_single,
                 [1] * len(_class_membership[0])
                 ]
            non_shared_values.append(_max_output_func(out))

    # Extract features
    travellers = _rides_row["indexes"]
    base_revenues = {num: _rides_row["individual_distances"][num] * _kmFare for num, pax in
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
    discounts = [[t if t[0] > _guaranteed_discount else (_guaranteed_discount, t[1])
                 for t in d] for d in discounts] # at least guaranteed discount
    discounts_indexes = list(itertools.product(*[range(len(t)) for t in discounts]))

    # Variable for tracking
    best = [0]*8

    discount_indexes: tuple[int]
    for discount_indexes in discounts_indexes:
        # Start with the effectively shared ride
        discount = [discounts[_num][_t] for _num, _t in enumerate(discount_indexes)]
        if any(t[0] > _max_discount for t in discount):
            continue
        eff_price = [_kmFare * (1 - t[0]) for t in discount]
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

        # Value of the future. First, shared option
        monetary_savings = [disc*dist*_kmFare/1000 for disc, dist in
                            zip(out[1], _rides_row["individual_distances"])]
        delta_utilities = [a - b for a, b in zip(monetary_savings, expected_time_utilities)]
        probability_potential = [_sigmoid(new + cur)
                                for new, cur in zip(delta_utilities, travellers_satisfaction)]
        future_value = ((np.prod(probability_potential) -
                        np.prod([_sigmoid(s) for s in travellers_satisfaction])) *
                        max_out) # shared rides
        future_value += sum(a*b for a, b in zip(
            [c - _sigmoid(d) for c, d in zip(probability_potential,travellers_satisfaction)],
            non_shared_values
        )) # single rides

        out += [future_value]
        out += [max_out + future_value]

        if max_out > best[-1]:
            membership_class_probability: list[list] = [[] for _ in range(len(travellers))]
            for num in range(len(travellers)):
                membership_class_probability[num] = [0]*len(_class_membership[0])
                for acceptable_discount in discounts[num][:(discount_indexes[num] + 1)]:
                    membership_class_probability[num][int(acceptable_discount[1])] += 1/_sample_size
            out.insert(-1, membership_class_probability)
            best = out.copy()

    return best


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def update_satisfaction(
        rides_row: pd.Series,
        predicted_class_distribution: dict,
        actual_class_distribution: dict,
        predicted_satisfaction: dict,
        actual_satisfaction: dict,
        vot_sample: dict,
        bs_levels: list,
        speed: float = 6,
        fare: float = 1.5
):
    _sample_size = int(len(vot_sample) / len(predicted_class_distribution[0].keys()))
    if fare > 1:
        km_fare: float = fare/1000
    else:
        km_fare: float = fare

    delta_perceived_times = [bs_levels[len(rides_row['indexes'])]*rides_row['individual_times'][num] -
                   rides_row['individual_distances'][num] / speed
                   for num in range(len(rides_row['indexes']))]

    monetary_savings = [disc * dist * km_fare / 1000 for disc, dist in
                        zip(rides_row['best_profit'][1], rides_row["individual_distances"])]
    # First, the expected satisfaction
    avg_vot = [sum(predicted_class_distribution[pax][vot_class] * vot
                        for vot, vot_class in vot_sample) / _sample_size
               for pax in rides_row['indexes']]
    expected_time_utilities = [avg_vot[num] / 3600 * delta_perceived_times[num]
                               for num in range(len(rides_row['indexes']))]
    delta_utilities = [a - b for a, b in zip(monetary_savings, expected_time_utilities)]

    new_predicted_satisfaction = {}
    for num, pax in enumerate(rides_row['indexes']):
        new_predicted_satisfaction[pax] = delta_utilities[num] + predicted_satisfaction[pax]

    # Second, actual satisfaction
    new_actual_satisfaction = {}
    for num, pax in enumerate(rides_row['indexes']):
        actual_avg_vot = np.mean([t[0] for t in vot_sample if t[1] == actual_class_distribution[pax]])
        expected_time_utility = actual_avg_vot/3600*delta_perceived_times[num]
        new_actual_satisfaction[pax] = monetary_savings[num] - expected_time_utility + actual_satisfaction[pax]

    return new_predicted_satisfaction, new_actual_satisfaction
