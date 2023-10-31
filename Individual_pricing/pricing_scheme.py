""" Script for analysis of the individual pricing """
import pandas as pd
from dotmap import DotMap
import datetime


def calculate_discount(u, p, d, b_t, b_s, t, t_d, b_d) -> float:
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
    @return: discount
    """
    return d - (1000 * u) / (p * b_t * b_s * (t + t_d * b_d))


def extract_travellers_data(
        databank: DotMap,
        params: DotMap
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
        'b_d': params["delay_value"],
        'pickup_time': datetime.timedelta(
            hours=t[1]["pickup_datetime"].time().hour,
            minutes=t[1]["pickup_datetime"].time().minute,
            seconds=t[1]["pickup_datetime"].time().second
        ).total_seconds()
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
        out.append(sum(times[(origin_index+1):(len(travellers)+1+destination_index)]))

    return out


def discount_row_func(
    row_rides: pd.Series,
    characteristics: dict
) -> list:
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
                t_d=sum(
                    row_rides["times"][:(row_rides["indexes_orig"].index(traveller)+1)]
                )-characteristics[traveller]["pickup_time"],
                b_d=characteristics[traveller]["b_d"]
            )
        )

    return out


def calculate_max_discount(
    databank: DotMap,
    travellers_characteristics: dict
) -> pd.DataFrame:
    rides = databank['exmas']['rides'].copy()
    rides["individual_times"] = rides.apply(lambda x:
                                            extract_individual_travel_times(x),
                                            axis=1)
    distances_dict = {t[0]: t[1]["dist"] for t in
                      databank['exmas']['requests'].iterrows()}
    rides["individual_distances"] = rides.apply(lambda x:
                                                [distances_dict[t] for t in x["indexes"]],
                                                axis=1)
    rides["max_discount"] = rides.apply(lambda x:
                                        discount_row_func(x, travellers_characteristics),
                                        axis=1)

    databank["recalibrated_rides"] = rides

    return databank
