""" Script for analysis of the individual pricing """
import itertools
from bisect import bisect_right
import os

import pandas as pd
import numpy as np
from dotmap import DotMap
from math import isnan
from tqdm import tqdm
from typing import Callable

from Individual_pricing.pricing_utils.product_distribution import ProductDistribution


def calculate_discount(u, p, d, b_t, b_s, t, t_d, b_d, shared) -> float:
    """
    Calculate discount to reach a certain log_level of shared utility
    @param u: utility of a non-shared ride
    @param p: price
    @param d: distance
    @param b_t: time sensitivity (VoT)
    @param b_s: sharing sensitivity (WtS)
    @param t: trip time
    @param t_d: delay length
    @param b_d: delay sensitivity
    @param shared: indicate whether a ride is shared
    @return: discount
    """
    dd = d / 1000
    if dd == 0 or not shared:
        return 0
    return 1 - u / (p * dd) + (b_t * b_s * (t + t_d * b_d)) / (p * dd)


def extract_travellers_data(
        databank: DotMap or dict,
        params: DotMap or dict
) -> dict:
    """
    Extract data for calculation of the maximal discount
    @param databank: the dotmap storing data in exmas
    @param params: parameters used in exmas
    @return: dictionary with passengers' characteristics
    """
    requests = databank['exmas']['requests'].copy()
    behavioural = databank['prob']['sampled_random_parameters'].copy().set_index('new_index')

    travellers_characteristics = {t[0]: {
        'u': t[1]['u'],
        'p': params['price'],
        'b_t': behavioural.loc[t[0], 'VoT'],
        'b_s': behavioural.loc[t[0], 'WtS'],
        'b_d': params["delay_value"]
    } for t in requests.iterrows()}

    return travellers_characteristics


def extract_individual_travel_times(
        row_rides: pd.Series
) -> list:
    """
    Function to be applied by rows to a dataframe rides
    @param row_rides: a row of the rides dataframe
    @return: travel times of travellers, respectively
    """
    travellers = row_rides["indexes"]
    times = row_rides["times"]
    out = []
    for traveller in travellers:
        origin_index = row_rides["indexes_orig"].index(traveller)
        destination_index = row_rides["indexes_dest"].index(traveller)
        out.append(sum(times[(origin_index + 1):(len(travellers) + 1 + destination_index)]))

    return out


def discount_row_func(
        row_rides: pd.Series,
        characteristics: dict
) -> list:
    """
    [row func] Calculate minimum discount per row of pd.Dataframe
    from the databank exmas rides
    @param row_rides: row
    @param characteristics: individual traits
    @return: updated row
    """
    travellers = row_rides["indexes"]
    out = []
    for no, traveller in enumerate(travellers):
        out.append(
            calculate_discount(
                u=characteristics[traveller]["u"],
                p=characteristics[traveller]["p"],
                d=row_rides["individual_distances"][no],
                b_t=characteristics[traveller]["b_t"],
                b_s=characteristics[traveller]["b_s"],
                t=row_rides["individual_times"][no],
                t_d=row_rides["delays"][no],
                b_d=characteristics[traveller]["b_d"],
                shared=(len(travellers) != 1)
            )
        )

    return out


def expand_rides(
        databank: DotMap or dict
) -> pd.DataFrame:
    """
    Add individual times and individual distances for rides df
    @param databank: the data bundle used in ExMAs
    @return: updated databank
    """
    rides = databank['exmas']['rides']
    rides["individual_times"] = rides.apply(extract_individual_travel_times,
                                            axis=1)
    distances_dict = {t[0]: t[1]["dist"] for t in
                      databank['exmas']['requests'].iterrows()}
    rides["individual_distances"] = rides.apply(lambda x:
                                                [distances_dict[t] for t in x["indexes"]],
                                                axis=1)
    return databank


def calculate_min_discount(
        databank: DotMap or dict,
        travellers_characteristics: dict
) -> DotMap:
    """
    Calculate minimum discount
    @param databank:
    @param travellers_characteristics:
    @return: updated databank
    """
    rides = databank['exmas']['rides']
    rides["min_discount"] = rides.apply(lambda x:
                                        [max(t, 0) for t in discount_row_func(x, travellers_characteristics)],
                                        axis=1)

    databank["exmas"]["recalibrated_rides"] = rides

    return databank


def calculate_profitability(
        databank: DotMap or dict,
        params: DotMap or dict
) -> DotMap or dict:
    """
    Calculate profitability of individual rides
    @param databank:
    @param params:
    @return: updated databank
    """

    def _base_row_revenue(row):
        if len(row["indexes"]) == 1:
            return row["individual_distances"][0] * params["price"]

        disc = params.get("true_discount") if not None else params["shared_discount"]
        out = sum(row["individual_distances"])
        out *= params["price"]
        out *= 1 - disc
        return out

    def _row_cost(row):
        return row["u_veh"] * params.get("operating_cost", 0.5)

    def _max_row_revenue(row):
        if len(row["indexes"]) == 1:
            return row["individual_distances"][0] * params["price"]

        out = 0
        for no, traveller in enumerate(row["indexes"]):
            disc = row["min_discount"][no]
            out += row["individual_distances"][no] * params["price"] * (1 - disc)
        return out

    rides = databank["exmas"]["recalibrated_rides"]
    rides["cost"] = rides.apply(lambda x: _row_cost(x), axis=1)

    rides["revenue_base"] = rides.apply(lambda x: _base_row_revenue(x), axis=1)
    rides["profit_base"] = rides["revenue_base"] - rides["cost"]

    rides["revenue_max"] = rides.apply(lambda x: _max_row_revenue(x), axis=1)
    rides["profit_max"] = rides["revenue_max"] - rides["cost"]

    rides["profitability_base"] = 1000 * rides["revenue_base"] / rides["cost"]
    rides["profitability_max"] = 1000 * rides["revenue_max"] / rides["cost"]

    for _n in ["profit_base", "profit_max",
               "profitability_base", "profitability_max"]:
        rides[_n] = rides[_n].apply(lambda x: 0 if isnan(x) else int(x))

    return databank


def row_expected_profitability(
        row_rides: pd.Series,
        params: DotMap or dict
) -> list:
    travellers = row_rides["indexes"]
    out = []
    for no, traveller in enumerate(travellers):
        return []


def prepare_samples(
        databank: DotMap or dict,
        sample_size: int = 100
) -> DotMap or dict:
    """
    Prepare samples of behavioural characteristics
    @param databank: list of outputs of the original ExMAS's format
    @param sample_size: size of sampled behavioural characteristics
    the longer, the improved accuracy of the approximation
    @return: list of dictionaries following the original ExMAS's format
    """
    from pricing_utils.product_distribution import ProductDistribution

    beta = ProductDistribution()
    beta.new_sample(
        distribution_type="multinormal",
        probs=[0.29, 0.28, 0.24, 0.19],
        means=[t / 3600 for t in [16.98, 14.02, 26.25, 7.78]],
        st_devs=[t / 3600 for t in [0.318, 0.201, 5.77, 1]],
        size=sample_size,
        seed=123
    )
    databank["prob"]["bt_sample"] = beta.sample.copy()

    databank["prob"]["bs_samples"] = {}
    for no_paxes, multiplier in zip([2, 3, 4, 5], [0.98, 1, 1.2, 2]):
        beta.remove_sample(0)
        beta.new_sample(
            distribution_type="multinormal",
            probs=[0.29, 0.28, 0.24, 0.19],
            means=[t * multiplier for t in [1.22, 1.135, 1.049, 1.18]],
            st_devs=[t * multiplier for t in [0.082, 0.071, 0.06, 0.076]],
            size=sample_size,
            seed=123
        )
        databank["prob"]["bs_samples"][no_paxes] = beta.sample.copy()

    return databank


def _row_sample_acceptable_disc(
        _rides_row: pd.Series,
        _times_non_shared: dict,
        _bs_samples: dict,
        _bt_sample: list,
        _interval: int,
        _price: float
) -> list:
    """
    Samples discounts for which clients accept rides
    at different discount levels
    @param _rides_row: the function works on rows of pd.Dataframe
    @param _times_non_shared: dictionary with individual rides
    obtained as dict(requests["ttrav"])
    @param _bs_samples: sampled willingness to share across
    the population
    @param _bt_sample: sampled value of time across the population
    @param _interval: once per how many items from the list
    should be returned for the final sample
    @param _price: price per metre
    @return: acceptable
    """
    no_travellers = len(_rides_row["indexes"])
    if no_travellers == 1:
        return []

    _bs = _bs_samples[no_travellers if no_travellers <= 5 else 5]
    out = []
    for no, trav in enumerate(_rides_row["indexes"]):
        out1 = [t * _rides_row["individual_times"][no] for t in _bs]
        out1 = [t - _times_non_shared[trav] for t in out1]
        out1 = sorted(a * b for a, b in itertools.product(out1, _bt_sample))
        out2 = []
        j = int(_interval / 2)
        while j < len(out1):
            out3 = out1[j] if out1[j] >= 0 else 0
            out2.append(out3 /
                        (_price * _rides_row["individual_distances"][no]))
            j += _interval
        out.append(out2)

    return out


def _row_maximise_profit(
        _rides_row: pd.Series,
        _one_shot: bool,
        _max_output_func: Callable[[list], float],
        _price: float = 0.0015,
        _probability_single: float = 1,
        _guaranteed_discount: float = 0.05,
        _min_acceptance: float = 0
):
    """
    Function to calculate the expected performance (or its derivatives)
     of a precalculated exmas ride
    --------
    @param _rides_row: row of exmas rides (potential combination + characteristics)
    @param _one_shot: scenario where if shared ride is not accepted by any of the
    co-travellers, all co-travellers drop out of the system
    @param _price: per-kilometre fare
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
    - max output function (by default, profitability)
    """
    no_travellers = len(_rides_row["indexes"])
    if no_travellers == 1:
        out = [_probability_single * _rides_row["veh_dist"] * _price * (1 - _guaranteed_discount),
                0,
                _rides_row["veh_dist"] * _price * (1 - _guaranteed_discount),
                [_probability_single],
                _rides_row["veh_dist"] / 1000 * _probability_single
        ]
        out += [_max_output_func(out)]
        return out

    discounts = _rides_row["accepted_discount"].copy()
    discounts = [[_guaranteed_discount] + t if t[0] > _guaranteed_discount else t for t in discounts]
    discounts = list(itertools.product(*discounts))

    # discounts = list(itertools.product(*_rides_row["accepted_discount"]))
    discounts = [[t if t > _guaranteed_discount else _guaranteed_discount
                  for t in discount] for discount in discounts]
    discounts = list(set(tuple(discount) for discount in discounts))
    base_revenues = {num: _rides_row["individual_distances"][num] * _price for num, t in
                     enumerate(_rides_row['indexes'])}
    best = [0, 0, 0, 0, 0, 0]
    # if _rides_row['indexes'] == [142, 133]:
    #     print('ee')
    for discount in discounts:
        """ For effectively shared ride """
        eff_price = [_price * (1 - t) for t in discount]
        revenue_shared = [a * b for a, b in
                          zip(_rides_row["individual_distances"], eff_price)]
        probability_shared = 1
        prob_ind = [0] * len(discount)

        # if _rides_row['indexes'] == [20, 39, 53]:
        #     print('eyy')

        for num, individual_disc in enumerate(discount):
            accepted_disc = _rides_row["accepted_discount"][num]
            prob_ind[num] = bisect_right(accepted_disc, individual_disc) / len(accepted_disc)
            probability_shared *= prob_ind[num]

        if probability_shared < _min_acceptance:
            continue

        if _one_shot:
            # out = [sum(revenue) * probability - cost, discount, sum(revenue), probability, cost]
            out = [sum(revenue_shared) * probability_shared,
                   discount,
                   sum(revenue_shared),
                   prob_ind,
                   _rides_row["veh_dist"] * probability_shared / 1000]
            max_out = _max_output_func(out)
            out += [max_out]
            if max_out > best[-1]:
                best = out.copy()
        else:
            # if (_rides_row['indexes'] == [43, 46]) & (sum(prob_ind) >= 1.3):
            #     print('ee')
            remaining_revenue = 0
            for num_trav in range(len(discount)):
                # First, if the P(X_j = 0)*r_j
                prob_not_trav = 1 - prob_ind[num_trav]
                remaining_revenue += prob_not_trav * base_revenues[num_trav]
                # Then, P(X_j = 1, \pi X_i = 0)*r_j*(1-\lambda)
                others_not = 1 - np.prod([t for num, t in enumerate(prob_ind) if num != num_trav])
                rev_discounted = base_revenues[num_trav] * (1 - _guaranteed_discount)
                remaining_revenue += (prob_ind[num_trav] * others_not) * rev_discounted

            out = [sum(revenue_shared) * probability_shared + remaining_revenue,
                   discount,
                   sum(revenue_shared),
                   prob_ind,
                   (_rides_row["veh_dist"] * probability_shared +
                    sum(_rides_row["individual_distances"]) * (1 - probability_shared)) / 1000]
            max_out = _max_output_func(out)
            out += [max_out]

            if max_out > best[-1]:
                best = out.copy()

    return best


def expected_profitability_function(
        databank: DotMap or dict,
        final_sample_size: int = 10,
        price: float = 0.0015,
        # cost_to_price_ratio: float = 0.3,
        max_func: Callable[[list], float] = lambda x: x[0]/x[4] if x[4] != 0 else 0,
        min_acceptance: float or None = None,
        one_shot: bool = False,
        guaranteed_discount: float = 0.1,
        speed: float = 6
) -> DotMap or dict:
    rides = databank["exmas"]["rides"]
    times_non_shared = dict(databank['exmas']['requests']['ttrav'])
    b_s = databank['prob']['bs_samples']
    b_t = databank['prob']['bt_sample']
    interval_size = int(len(b_s[2]) * len(b_t) / final_sample_size)
    # interval_size = int(len(b_s[2]) * len(b_t) / (2*final_sample_size))

    tqdm.pandas()

    rides["accepted_discount"] = rides.progress_apply(
        _row_sample_acceptable_disc,
        axis=1,
        _times_non_shared=times_non_shared,
        _bs_samples=b_s,
        _bt_sample=b_t,
        _interval=interval_size,
        _price=price
    )

    rides["veh_dist"] = rides["u_veh"] * speed

    rides["best_profit"] = rides.progress_apply(_row_maximise_profit,
                                                axis=1,
                                                _price=price,
                                                _one_shot=one_shot,
                                                _max_output_func=max_func,
                                                _guaranteed_discount=guaranteed_discount,
                                                _min_acceptance=min_acceptance
                                                )

    return databank


def profitability_measures(
        databank: DotMap or dict,
        op_costs: list[float] or tuple[float] = (0.2, 0.3, 0.4, 0.5, 0.6)
):
    rides = databank["exmas"]["rides"]
    rides["expected_revenue"] = rides["best_profit"].apply(lambda x: x[0])
    rides["acceptance_prob"] = rides["best_profit"].apply(lambda x: np.prod(x[3]))

    rides["profitability"] = rides["best_profit"].apply(lambda x: x[5] * len(x[3]))
    objectives = ["expected_revenue", "profitability"]

    for op_cost in op_costs:
        rides["expected_cost_" + str(int(100 * op_cost))] = rides["best_profit"].apply(lambda x: op_cost * x[4])
        rides["expected_profit_" + str(int(100 * op_cost))] = rides["expected_revenue"] \
                                                              - rides["expected_cost_" + str(int(100 * op_cost))]
        rides["expected_profit_int_" + str(int(100 * op_cost))] = rides["expected_profit_"
                                                                        + str(int(100 * op_cost))].apply(
            lambda x: int(1000 * x))
        objectives += ["expected_profit_int_" + str(int(100 * op_cost))]

    databank["exmas"]["recalibrated_rides"] = rides.copy()
    databank["exmas"]["rides"] = rides
    databank["exmas"]["objectives"] = objectives

    return databank


def calculate_delta_utility(
        discount: float,
        price: float,
        ns_trip_dist: float,
        vot: float,
        wts: float,
        travel_time_ns: float,
        travel_time_s: float
):
    out = discount * price * ns_trip_dist
    out -= vot * wts * (travel_time_s - travel_time_ns)
    return out


def check_prob_if_accepted(
        row_rides: pd.Series,
        discount: float,
        total: bool = True
):
    if total:
        if len(row_rides["indexes"]) == 1:
            return 1

        out = 1
        for pax_disc in row_rides["accepted_discount"]:
            out *= bisect_right(pax_disc, discount)
            out /= len(pax_disc)

        return out

    if len(row_rides["indexes"]) == 1:
        return [1]

    out = []
    for pax_disc in row_rides["accepted_discount"]:
        out += [bisect_right(pax_disc, discount) / len(pax_disc)]

    return out


def _expected_flat_measures(
        vector_probs: pd.Series or list,
        shared_dist: float,
        ind_dists: pd.Series or list,
        price: float,
        sharing_disc: float,
        guaranteed_disc: float
):
    if len(vector_probs) == 1:
        return [price * shared_dist/1000, shared_dist/1000]

    prob_shared = np.prod(vector_probs)
    # if shared
    rev = prob_shared * sum(ind_dists) * price * (1 - sharing_disc) / 1000

    # if not shared
    for pax in range(len(vector_probs)):
        prob_not_trav = 1 - vector_probs[pax]
        rev += ind_dists[pax] * prob_not_trav * price / 1000

        others_not = 1 - np.prod(vector_probs[:pax] + vector_probs[(pax + 1):])
        rev += ind_dists[pax] * others_not * vector_probs[pax] * (1 - guaranteed_disc) * price / 1000

    dist = prob_shared * shared_dist / 1000 + (1 - prob_shared) * sum(ind_dists) / 1000

    return [rev, dist]


def check_percentiles_distribution(ab, perc, multiplier=1):
    beta = ProductDistribution()
    if ab:
        beta.new_sample(
            distribution_type="multinormal",
            probs=[0.29, 0.28, 0.24, 0.19],
            means=[t / 3600 for t in [16.98, 14.02, 26.25, 7.78]],
            st_devs=[t / 3600 for t in [0.318, 0.201, 5.77, 1]],
            size=1000,
            seed=123
        )
    else:
        beta.new_sample(
            distribution_type="multinormal",
            probs=[0.29, 0.28, 0.24, 0.19],
            means=[t * multiplier for t in [1.22, 1.135, 1.049, 1.18]],
            st_devs=[t * multiplier for t in [0.082, 0.071, 0.06, 0.076]],
            size=1000,
            seed=123
        )
    return np.percentile(beta.sample, perc)


def path_joiner(path1: str, path2: str):
    return os.path.join(path1, path2)
