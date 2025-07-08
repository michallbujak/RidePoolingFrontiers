""" Pricing baselines """
import argparse
from pathlib import Path
import pickle
from typing import Callable
from itertools import product
from bisect import bisect

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from Individual_pricing.matching import matching_function_light

parser = argparse.ArgumentParser()
parser.add_argument("--data-pickle", type=str, required=True)
parser.add_argument("--sample-size", type=int, default=10)
args = parser.parse_args()
print(args)


x = 0
def baseline_jiao(
        row_rides: pd.Series,
        sample_size: int,
        max_func: Callable[[list], float] = lambda x: x[0]/x[4] if x[4] != 0 else 0,
        m_fare: float = 0.0015,
        guaranteed_discount: float = 0.05,
        max_discount: float = 0.4,
        discount_interval: float = 0.05,
        mean_vot: float = 0.0046,
        mean_pfs: float = 1.1246,
        avg_speed: float = 6
):
    """
    Pricing based on method proposed in
    'Incentivizing shared rides in e-hailing markets:
    Dynamic discounting' by Guipeng Jiao & Mohsen Ramezani
    adopted to study's assumptions
    """
    if len(row_rides['indexes']) == 1:
        out = [row_rides["veh_dist"] * m_fare * (1 - guaranteed_discount),
               0,
               row_rides["veh_dist"] * m_fare * (1 - guaranteed_discount),
               [1],
               row_rides["veh_dist"] / 1000 * 1
               ]
        out += [max_func(out)]
        return out

    if len(row_rides['indexes']) > 2:
        return [0]*6


    best_vector: list = [0]*6
    discount_space = np.arange(guaranteed_discount,
                               max_discount + discount_interval,
                               discount_interval)
    discount_space = list(round(t, 2) for t in discount_space)

    # We adopt our utility formulation for a fair comparison
    u_private = [-(m_fare + mean_vot / avg_speed) * _d
                 for _d in row_rides['individual_distances']]
    forced_prices = [(1 - guaranteed_discount) * _d * m_fare for _d in row_rides['individual_distances']]
    rejected_prices = [_d * m_fare for _d in row_rides['individual_distances']]

    for discounts in product(discount_space, discount_space):
        # Estimate probability based on means and Logit model
        u_pool = [(_l - 1)*m_fare*_d - mean_pfs*mean_vot*_t
                  for _l, _d, _t in zip(discounts, row_rides['individual_distances'],
                                        row_rides['individual_times'])]
        logit_probs = [np.exp(b)/(np.exp(a) + np.exp(b)) for a, b in zip(u_private, u_pool)]
        actual_probs = [bisect(t, _l)/sample_size for t, _l
                              in zip(row_rides['accepted_discount'], discounts)]

        # Now, proceed to the expected profitability (instead of profit
        # to adjust as consistent with the main framework)
        shared_prices = [(1-_l)*_d*m_fare for _l, _d in zip(discounts, row_rides['individual_distances'])]

        ex_revenue, ex_dist = dict(), dict()
        for method, probs in zip(['logit', 'actual'], [logit_probs, actual_probs]):
            ex_revenue[method] = np.prod(probs)*sum(shared_prices)
            ex_revenue[method] += (1 - probs[0])*probs[1]*(rejected_prices[0]+forced_prices[1])
            ex_revenue[method] += (1 - probs[1])*probs[0]*(rejected_prices[1]+forced_prices[0])
            ex_revenue[method] += (1 - probs[0])*(1 - probs[0])*sum(rejected_prices)

            ex_dist[method] = np.prod(probs)*row_rides['veh_dist']
            ex_dist[method] += (1 - probs[0])*probs[1]*sum(row_rides['individual_distances'])
            ex_dist[method] += (1 - probs[1])*probs[0]*sum(row_rides['individual_distances'])
            ex_dist[method] += (1 - probs[0])*(1 - probs[0])*sum(row_rides['individual_distances'])
            ex_dist[method] /= 1000

        performance_vector = [
            ex_revenue['logit'],
            discounts,
            sum(shared_prices),
            actual_probs,
            ex_dist['logit']
        ]
        performance_vector += [max_func(performance_vector)]
        performance_vector[0] = ex_revenue['actual']
        performance_vector[4] = ex_dist['actual']

        if performance_vector[-1] > best_vector[-1]:
            best_vector = performance_vector.copy()

    return best_vector


def baseline_karaenke(
        row_rides: pd.Series,
        sample_size: int,
        max_func: Callable[[list], float] = lambda x: x[0] / x[4] if x[4] != 0 else 0,
        m_fare: float = 0.0015,
        guaranteed_discount: float = 0.05,
        max_discount: float = 0.4,
        discount_interval: float = 0.05,
        mean_vot: float = 0.0046,
        mean_pfs: float = 1.1246,
        avg_speed: float = 6
):
    """
     Pricing scheme adopted to the study according to
     'On the benefits of ex-post pricing for ride-pooling'
     by Paul Karaenke, Maximilian Schiffer, Stefan Waldherr
    """
    if len(row_rides['indexes']) == 1:
        out = [row_rides["veh_dist"] * m_fare * (1 - guaranteed_discount),
               0,
               row_rides["veh_dist"] * m_fare * (1 - guaranteed_discount),
               [1],
               row_rides["veh_dist"] / 1000 * 1
               ]
        out += [max_func(out)]
        return out

    if len(row_rides['indexes']) > 2:
        return [0]*6

    # First, calculate 5-point distribution of VoT
    weights = [0.19, 0.28, 0.29, 0.24]
    means = [7.78, 14.02, 16.98, 26.25]
    std_devs = [1, 0.201, 0.318, 5.77]

    def mixture_cdf(x):
        return sum(w * norm.cdf(x, loc=m, scale=s) for w, m, s in zip(weights, means, std_devs))

    def find_percentile(p):
        return brentq(lambda x: mixture_cdf(x) - p, 0, 40)

    percentiles = [0.1, 0.9]
    vot_end_points: list = [find_percentile(p) for p in percentiles]
    vot_points: list = np.linspace(*vot_end_points, 5).tolist()

    # Calculate discounts based on a 5% utility gain:
    # U^{ns} = -\rho d - \beta_t t^{ns}
    # U^s = -(1 - \lambda) \rho d - \beta_t \beta_s t^s
    # U^s = 0.95 U^{ns}
    # \lambda = (0.05 \rho d - \beta_t(t^{ns} - \beta_s t^s))/(\rho d)
    discounts = [0, 0]
    for vot1, vot2 in product(vot_points, vot_points):
        vot_vector = [vot1/3600, vot2/3600]
        for pax in range(2):
            nominator = ((0.05*m_fare*row_rides['individual_distances'][pax]) -
                         vot_vector[pax]*(row_rides['individual_distances'][pax]/avg_speed -
                                                mean_pfs*row_rides['individual_times'][pax]))
            denominator = m_fare*row_rides['individual_distances'][pax]
            discounts[pax] += nominator/denominator
    discounts = [t/(len(vot_points)*len(vot_points)) for t in discounts]
    discounts = [t if t >= guaranteed_discount else guaranteed_discount for t in discounts]
    discounts = [t if t <= max_discount else max_discount for t in discounts]

    shared_prices = [(1 - _l) * _d * m_fare for _l, _d in zip(discounts, row_rides['individual_distances'])]
    forced_prices = [(1 - guaranteed_discount) * _d * m_fare for _d in row_rides['individual_distances']]
    rejected_prices = [_d * m_fare for _d in row_rides['individual_distances']]

    actual_probs = [bisect(t, _l) / sample_size for t, _l
                    in zip(row_rides['accepted_discount'], discounts)]

    ex_revenue = np.prod(actual_probs) * sum(shared_prices)
    ex_revenue += (1 - actual_probs[0]) * actual_probs[1] * (actual_probs[0] + actual_probs[1])
    ex_revenue += (1 - actual_probs[1]) * actual_probs[0] * (rejected_prices[1] + forced_prices[0])
    ex_revenue += (1 - actual_probs[0]) * (1 - actual_probs[0]) * sum(rejected_prices)

    ex_dist = np.prod(actual_probs) * row_rides['veh_dist']
    ex_dist += (1 - actual_probs[0]) * actual_probs[1] * sum(row_rides['individual_distances'])
    ex_dist += (1 - actual_probs[1]) * actual_probs[0] * sum(row_rides['individual_distances'])
    ex_dist += (1 - actual_probs[0]) * (1 - actual_probs[0]) * sum(row_rides['individual_distances'])
    ex_dist /= 1000

    performance_vector = [
        ex_revenue,
        discounts,
        sum(shared_prices),
        actual_probs,
        ex_dist
    ]
    performance_vector += [max_func(performance_vector)]

    return performance_vector


def baseline_distance(
        row_rides: pd.Series,
        sample_size: int,
        max_func: Callable[[list], float] = lambda x: x[0] / x[4] if x[4] != 0 else 0,
        m_fare: float = 0.0015,
        guaranteed_discount: float = 0.05,
        max_discount: float = 0.4,
        discount_interval: float = 0.05,
        mean_vot: float = 0.0046,
        mean_pfs: float = 1.1246,
        avg_speed: float = 6
):
    """
    A natural baseline, where travellers get a discount
    proportional to the additional distance they experience with pooling
    """
    if len(row_rides['indexes']) == 1:
        out = [row_rides["veh_dist"] * m_fare * (1 - guaranteed_discount),
               0,
               row_rides["veh_dist"] * m_fare * (1 - guaranteed_discount),
               [1],
               row_rides["veh_dist"] / 1000 * 1
               ]
        out += [max_func(out)]
        return out

    additional_distance = [a - b/avg_speed for a, b in
                           zip(row_rides['individual_distances'], row_rides['individual_times'])]
    relative_add_distance = [a/b for a, b in
                             zip(additional_distance, row_rides['individual_distances'])]
    relative_add_distance = [t if t < 0.25 else 0.25 for t in relative_add_distance]
    discounts = [guaranteed_discount + (max_discount-guaranteed_discount)*4*t
                 for t in relative_add_distance]

    actual_probs = [bisect(t, _l) / sample_size for t, _l
                    in zip(row_rides['accepted_discount'], discounts)]

    shared_prices = [(1 - _l) * _d * m_fare for _l, _d in zip(discounts, row_rides['individual_distances'])]
    forced_prices = [(1 - guaranteed_discount) * _d * m_fare for _d in row_rides['individual_distances']]
    rejected_prices = [_d * m_fare for _d in row_rides['individual_distances']]

    ex_revenue = np.prod(actual_probs) * sum(shared_prices)
    ex_revenue += (1 - actual_probs[0]) * actual_probs[1] * (actual_probs[0] + actual_probs[1])
    ex_revenue += (1 - actual_probs[1]) * actual_probs[0] * (rejected_prices[1] + forced_prices[0])
    ex_revenue += (1 - actual_probs[0]) * (1 - actual_probs[0]) * sum(rejected_prices)

    ex_dist = np.prod(actual_probs) * row_rides['veh_dist']
    ex_dist += (1 - actual_probs[0]) * actual_probs[1] * sum(row_rides['individual_distances'])
    ex_dist += (1 - actual_probs[1]) * actual_probs[0] * sum(row_rides['individual_distances'])
    ex_dist += (1 - actual_probs[0]) * (1 - actual_probs[0]) * sum(row_rides['individual_distances'])
    ex_dist /= 1000

    performance_vector = [
        ex_revenue,
        discounts,
        sum(shared_prices),
        actual_probs,
        ex_dist
    ]
    performance_vector += [max_func(performance_vector)]

    return performance_vector



with open(args.data_pickle, 'rb') as _f:
    data = pickle.load(_f)[0]

rides = data['exmas']['rides']

for baseline_name, baseline_method in zip(['jiao', 'karaenke', 'distance'],
                                          [baseline_jiao, baseline_karaenke, baseline_distance]):
    rides['baseline_' + baseline_name] = rides.apply(
        baseline_method,
        sample_size=args.sample_size,
        axis=1
    )

    rides['profitability_' + baseline_name + '_est'] = rides.apply(
        lambda x: x['baseline_' + baseline_name][-1]*len(x['indexes']),
        axis=1
    )

    data['exmas']['schedules'][baseline_name] = matching_function_light(
        _rides=rides,
        _requests=data['exmas']['requests'],
        _objective='profitability_' + baseline_name + '_est',
        _min_max='max'
    )
    data['exmas']['schedules'][baseline_name]['profitability_' + baseline_name + '_actual'] = (
        data['exmas']['schedules'][baseline_name]['baseline_' + baseline_name].apply(
            lambda x: x[0] * len(x[3]) / x[4] if x[4] != 0 else 0
        )
    )

path = Path(args.data_pickle)
with open(str(path.parent) + '/' + path.name.split('.')[0] + '_baselines.pickle', 'wb') as _file:
    pickle.dump(data, _file)

