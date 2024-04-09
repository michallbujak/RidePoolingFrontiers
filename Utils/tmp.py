import pickle
import random
import bisect
import scipy.stats as ss
import numpy as np
import pandas as pd
from dotmap import DotMap
from tqdm import tqdm

from Individual_pricing.pricing_functions import extract_individual_travel_times
from visualising_functions import exmas_formula_s_explicit, exmas_formula_ns_explicit

with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Miscellaneous_scripts\data\22-03-24\147_homo_22-03-24.obj",
          "rb") as file:
    data = pickle.load(file)

schedule = data[0]["sblts"]["schedule"]
schedule["individual_times"] = schedule.apply(extract_individual_travel_times, axis=1)
requests = data[0]["sblts"]["requests"]
requested_distances = requests["dist"].to_dict()


multinormal_probs = (0.29, 0.57, 0.81, 1)
multinormal_args = (
    ((16.98 / 3600, 1.22), (0.31765 / 3600, 0.0815)),
    ((14.02 / 3600, 1.135), (0.2058 / 3600, 0.07056)),
    ((26.25 / 3600, 1.049), (5.7765 / 3600, 0.06027)),
    ((7.78 / 3600, 1.18), (1 / 3600, 0.07626))
)


def acceptance_row(
        schedule_row: pd.Series,
        req_dist: dict,
        vot_wts: pd.DataFrame
):
    price = 1.5
    avg_speed = 6
    discount = 0.3

    out = []
    acc = 1
    if len(schedule_row["indexes"]) == 1:
        return [1], acc

    for no, traveller in enumerate(schedule_row["indexes"]):
        ns_utility = exmas_formula_ns_explicit(
            _price=price,
            _distance=req_dist[traveller],
            _vot=vot_wts.loc[traveller, "vot"],
            _travel_time=req_dist[traveller] / avg_speed
        )
        s_utility = exmas_formula_s_explicit(
            _price=price,
            _discount=discount,
            _distance=req_dist[traveller],
            _vot=vot_wts.loc[traveller, "vot"],
            _wts=vot_wts.loc[traveller, "wts"],
            _total_time=schedule_row["individual_times"][no]
        )
        s_utility += np.random.normal() + np.random.normal(0, 0.1)
        if ns_utility < s_utility:
            out.append(0)
            acc = 0
        else:
            out.append(1)

    return out, acc


def mixed_discrete_norm_distribution(probs, arguments, with_index=True):
    def internal_function(*X):
        z = random.random()
        index = bisect.bisect(probs, z)
        if with_index:
            return [ss.norm.ppf(x, loc=mean, scale=std) for x, mean, std in
                    zip(X, arguments[index][0], arguments[index][1])] + [
                index]
        else:
            return [ss.norm.ppf(x, loc=mean, scale=std) for x, mean, std in
                    zip(X, arguments[index][0], arguments[index][1])]

    return internal_function


gen_func = mixed_discrete_norm_distribution(
    probs=multinormal_probs,
    arguments=multinormal_args
)

satisfaction = {
    "people_sharing": 0,
    "people_satisfied": 0,
    "people_realised": 0,
    "shared_rides": 0,
    "rides_realised": 0,
    "2": 0,
    "2_realised": 0,
    "3": 0,
    "3_realised": 0,
    "4": 0,
    "4_realised": 0,
}

pbar = tqdm(total=1000)
for rep in range(1000):
    sample_from_interval = np.random.random([147, 2])
    individual_params = pd.DataFrame([gen_func(*sample_from_interval[j, :])
                                      for j in range(len(sample_from_interval))])
    individual_params.columns = ["vot", 'wts', 'class']
    o = schedule.apply(acceptance_row,
                       vot_wts=individual_params,
                       req_dist=requested_distances,
                       axis=1)
    schedule["accepted"] = [t[0] for t in o]
    schedule["realised"] = [t[1] for t in o]

    shared = schedule.loc[[len(t) > 1 for t in schedule["indexes"]]]
    acceptance = [a for b in shared["accepted"] for a in b]
    satisfaction['people_sharing'] += len(acceptance)
    satisfaction['people_satisfied'] += sum(acceptance)
    satisfaction['people_realised'] += sum(
        shared.apply(lambda x: len(x["accepted"]) * x["realised"], axis=1)
    )
    satisfaction["shared_rides"] += len(shared)
    satisfaction["rides_realised"] += len(shared.loc[shared["realised"] == 1])

    for no in [2, 3, 4]:
        if no in [2, 3]:
            shared_tmp = schedule.loc[[len(t) == no for t in schedule["indexes"]]]
        else:
            shared_tmp = schedule.loc[[len(t) >= no for t in schedule["indexes"]]]
        satisfaction[str(no)] += len(shared_tmp)
        satisfaction[str(no) + "_realised"] += len(shared_tmp.loc[shared_tmp["realised"] == 1])

    pbar.update(1)

print(satisfaction)
print(satisfaction)
