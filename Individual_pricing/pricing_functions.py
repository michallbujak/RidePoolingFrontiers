""" Script for analysis of the individual pricing """
import itertools
from bisect import bisect_right

import pandas as pd
from dotmap import DotMap
from math import isnan

from Individual_pricing.pricing_utils.product_distribution import ProductDistribution


def calculate_discount(u, p, d, b_t, b_s, t, t_d, b_d, shared) -> float:
    """
    Calculate discount to reach a certain level of shared utility
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
    for no_paxes, multiplier in zip([2, 3, 4, 5], [0.95, 1, 1.1, 1.2, 2]):
        beta.remove_sample(0)
        beta.new_sample(
            distribution_type="multinormal",
            probs=[0.29, 0.28, 0.24, 0.19],
            means=[t * multiplier for t in [1.22, 1.135, 1.049, 1.18]],
            st_devs=[t / 3600 for t in [0.318, 0.201, 5.77, 1]],
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
    @param _times_non_shared: dictionary with iniduvidal rides
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
            out2.append(out1[j] /
                        (_price * _rides_row["individual_distances"][no]))
            j += _interval
        out.append(out2)

    return out


def _row_maximise_profit(
        _rides_row: pd.Series,
        _price: float = 0.0015,
        _cost_to_price_ratio: float = 0.3,
        _sample_size: int = 10
):
    no_travellers = len(_rides_row["indexes"])
    if no_travellers == 1:
        return _rides_row["veh_dist"]*_price*(1-_cost_to_price_ratio)

    discounts = list(itertools.product(*_rides_row["accepted_discount"]))
    best = [0, 0, 0]
    for discount in discounts:
        eff_price = [_price*(1 - t) for t in discount]
        revenue = [a * b for a, b in
                   zip(_rides_row["individual_distances"], eff_price)]
        cost = _rides_row["veh_dist"]*_price*_cost_to_price_ratio
        probability = 1
        for num, ind_disc in enumerate(discount):
            sample = _rides_row["accepted_discount"][num]
            probability *= bisect_right(sample, ind_disc)
            probability /= _sample_size

        out = [sum(revenue) * probability - cost, discount, cost]
        if out[0] > best[0]:
            best = out.copy()

    return best


# def _row_recalculate_utility(
#         row_rides: pd.Series,
#         times_non_shared: dict,
#         price: float
# ) -> list:
#     if len(row_rides["indexes"]) == 1:
#         return row_rides["u_pax"]
#
#     utilities = []
#
#     for num, traveller in enumerate(row_rides["indexes"]):
#         discount = row_rides["best_profit"][1][num]
#         distance = row_rides["individual_distances"][num]
#         time_ns = times_non_shared[traveller]
#         time_s = row_rides["individual_times"][num]



def calculate_expected_profitability(
        databank: DotMap or dict,
        final_sample_size: int = 10,
        price: float = 0.0015,
        cost_to_price_ratio: float = 0.3,
        speed: float = 6
) -> DotMap or dict:
    rides = databank["exmas"]["rides"]
    times_non_shared = dict(databank['exmas']['requests']['ttrav'])
    b_s = databank['prob']['bs_samples']
    b_t = databank['prob']['bt_sample']
    interval_size = int(len(b_s[2]) * len(b_t) / final_sample_size)
    rides["accepted_discount"] = rides.apply(
        _row_sample_acceptable_disc,
        axis=1,
        _times_non_shared=times_non_shared,
        _bs_samples=b_s,
        _bt_sample=b_t,
        _interval=interval_size,
        _price=price
    )

    # def foo(_arg):
    #     if len(_arg) == 0:
    #         return []
    #     return sorted([a for b in _arg for a in b])

    # rides["discount_threshold"] = rides["accepted_discount"].apply(foo)

    rides["veh_dist"] = rides["u_veh"]*speed
    rides["best_profit"] = rides.apply(_row_maximise_profit,
                                       axis=1,
                                       _price=price,
                                       _cost_to_price_ratio=cost_to_price_ratio,
                                       _sample_size=final_sample_size)
    rides["max_profit"] = rides["best_profit"].apply(lambda x: x[0] * price)
    rides["max_profit_int"] = rides["best_profit"].apply(lambda x: int(1000 * x))

    databank["exmas"]["recalibrated_rides"] = rides.copy()

    return databank


# def maximum_profit(
#         databank: DotMap or dict,
#         cost_to_price_ratio: float = 0.3
# ) -> DotMap or dict:
#     rides = databank["exmas"]["rides"]
#     rides["expected_profit"] = rides["max_expected_profit"].apply(lambda x: x[0])
#     rides["expected_profit"] -= rides["u_veh"] * cost_to_price_ratio
#
#     return databank
