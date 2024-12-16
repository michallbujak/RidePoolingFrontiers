""" Function to support the dynamic pricing algorithm """
import secrets
from typing import Callable, Tuple, Any
import itertools
from bisect import bisect_right
from unittest.mock import inplace

import numpy as np
import pandas as pd
from tqdm import tqdm


def prepare_samples(
        sample_size: int,
        means: list or tuple,
        st_devs: list or tuple,
        seed: int = 123,
        descending: bool = False
):
    """
    Prepare behavioural samples to create a discrete distribution
    instead of the continuous normal
    :param sample_size: numer of samples per class
    :param means: means per each class
    :param st_devs: respective standard deviations
    :param seed: seed for reproducibility
    :param descending: return values in descending order
    :return: discrete behavioural samples
    """
    rng = np.random.default_rng(secrets.randbits(seed))
    out = []

    for subpop_num, mean in enumerate(means):
        pop_sample = rng.normal(
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
        _min_acceptance: float = 0
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
    --------
    @return vector comprising 6 main characteristics when applied discount maximising
    the expected revenue:
    - expected revenue
    - vector of individual discounts
    - revenue from the shared ride if accepted
    - vector of probabilities that individuals accept the shared ride
    - expected distance
    - probability of acceptance when in certain class: [t1 \in C1, t1 \in C2,...], [t2 \in C1, t2 \in C2, ...]
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
        speed: float = 6
) -> pd.DataFrame:

    tqdm.pandas(desc="Accepted discount calculations")

    rides["accepted_discount"] = rides.progress_apply(
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

    tqdm.pandas(desc="Discount optimisation")
    rides["best_profit"] = rides.progress_apply(row_maximise_profit_bayes,
                                                axis=1,
                                                _class_membership=class_membership,
                                                _sample_size=int(len(bt_sample)/len(class_membership[0].keys())),
                                                _fare=fare,
                                                _max_output_func=objective_func,
                                                _guaranteed_discount=guaranteed_discount,
                                                _min_acceptance=min_acceptance
                                                )

    rides["objective"] = rides['best_profit'].apply(lambda x: x[-1])

    return rides


def bayesian_vot_updated(
        decision: bool,
        pax_id: int,
        apriori_distribution: dict,
        conditional_probs: list
):
    """
    Update dictionary with class membership probabilities according to the decision.
    Calculates for a single traveller.
    :param decision:
    :param pax_id:
    :param apriori_distribution:
    :param conditional_probs:
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

    return apriori_distribution


def aggregate_daily_results(
        day_results: dict,
        decisions: list,
        fare: float,
        guaranteed_discount: float = 0.05,
        speed: float = 6
):
    """
    Aggregate results after each day to track system's evolution.
    :param day_results: results from a daily run
    :param decisions: list of decisions: whether a shared ride is realised or not
    :param fare: per-kilometre price
    :param guaranteed_discount: discount for traveller who accepted sharing mode
    and their co-traveller rejected
    :param a constant speed (assumption, input)
    :return: table with results
    """
    results = pd.Series()
    schedule = day_results['schedules']['objective']
    results['TravellersNo'] = len(day_results['requests'])
    results['Objective'] = sum(schedule['objective'])
    schedule_sharing = schedule.loc[schedule['indexes'].apply(lambda x: len(x) >= 2)]
    schedule_sharing = schedule_sharing.reset_index(inplace=False, drop=True)
    results['SharingTravellerOffer'] = sum(schedule_sharing['indexes'].apply(len))
    results['SharedRidesNo'] = len(schedule_sharing)
    results['ExpectedSharingFraction'] = sum(schedule_sharing['indexes'].apply(len))/len(day_results['requests'])
    results['ExpectedRevenue'] = sum(schedule['best_profit'].apply(lambda x: x[0]))
    results['ExpectedDistance'] = sum(schedule['best_profit'].apply(lambda x: x[4]))
    results['ExpectedSharingRevenue'] = sum(schedule_sharing['best_profit'].apply(lambda x: x[0]))
    results['ExpectedSharingDistance'] = sum(schedule_sharing['best_profit'].apply(lambda x: x[4]))


    actualSharingRevenue: float = 0
    actualSharingDistance: float = 0
    actualAcceptanceRate: float = 0
    realisedSharedRides: int = 0
    for decision_indicator, shared_ride in schedule_sharing.iterrows():
        # if the share ride is realised
        if all(decisions[decision_indicator]):
            actualSharingRevenue += shared_ride['best_profit'][2]
            actualSharingDistance += shared_ride['u_veh']*speed/1000
            actualAcceptanceRate += len(shared_ride['indexes'])
            realisedSharedRides += 1
        else:
            actualSharingDistance += sum(shared_ride['individual_distances'])/1000
            for pax_no, pax in enumerate(shared_ride['indexes']):
                decision = decisions[decision_indicator][pax_no]
                if decision:
                    actualAcceptanceRate += 1
                    actualSharingRevenue += (shared_ride['individual_distances'][pax_no]*
                                             fare*(1-guaranteed_discount)/1000)
                else:
                    actualSharingRevenue += shared_ride['individual_distances'][pax_no]/1000
    results['ActualSharingRevenue'] = actualSharingRevenue
    results['ActualRevenue'] = (actualSharingRevenue + (1-guaranteed_discount)*
                                 fare*sum(schedule_sharing['u_veh'])*speed/1000)
    results['ActualSharingDistance'] = actualSharingDistance
    results['ActualDistance'] = actualSharingDistance + sum(schedule_sharing['u_veh'])*speed/1000
    results['RealisedSharedRides'] = realisedSharedRides
    results['ActualAcceptanceRate'] = actualAcceptanceRate/results['SharingTravellerOffer']

    return results
