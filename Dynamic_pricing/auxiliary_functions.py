""" Function to support the dynamic pricing algorithm """
import secrets
from typing import Callable

import numpy as np
import pandas as pd
import tqdm

from Individual_pricing.posteriori_analysis_old import price
from Individual_pricing.pricing_functions import row_maximise_profit


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
        pax_out /= (_fare * _rides_row["individual_distances"][no])
        out.append(pax_out)

    return out


def row_maximise_profit_bayes(
        _rides_row: pd.Series,
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


def expected_profit_function(
        rides: pd.DataFrame,
        requests: pd.DataFrame,
        bt_sample: np.array,
        bs_levels: list[float] or dict,
        objective_func: Callable[[list], float] = lambda x: x[0] - x[4],
        min_acceptance: float or None = None,
        guaranteed_discount: float = 0.1,
        fare: float = 0.0015,
        speed: float = 6
) -> pd.DataFrame:

    times_non_shared = dict(requests['ttrav'])
    tqdm.pandas()

    rides["accepted_discount"] = rides.progress_apply(
        row_acceptable_discount_bayes,
        axis=1,
        _times_non_shared=times_non_shared,
        _bs_samples=bs_levels,
        _bt_sample=bt_sample,
        _interval=1,
        _price=fare
    )

    rides["veh_dist"] = rides["u_veh"] * speed

    rides["best_profit"] = rides.progress_apply(row_maximise_profit,
                                                axis=1,
                                                _price=fare,
                                                _one_shot=False,
                                                _max_output_func=objective_func,
                                                _guaranteed_discount=guaranteed_discount,
                                                _min_acceptance=min_acceptance
                                                )

    return rides




