import pandas as pd
from dotmap import DotMap

from ExMAS.probabilistic_exmas import match


def matching_function(
    databank: DotMap or dict,
    params: DotMap or dict,
    objectives: list or None = None,
    min_max: str = "min"
):
    if objectives is None:
        objectives = databank["exmas"]["objectives"].copy()

    rides = databank["exmas"]["recalibrated_rides"]
    requests = databank["exmas"]["requests"]

    schedules = {}

    for objective in objectives:
        params.matching_obj = objective
        selected = match(
            im=rides,
            r=requests,
            params=params,
            min_max=min_max
        )

        rides["selected_" + objective] = pd.Series(selected.copy())

        schedules[objective] = rides.loc[[bool(t) for t in rides["selected_" + objective]]].copy()

        req_ride_dict = dict()
        for i, trips in schedules[objective].iterrows():
            for trip in trips.indexes:
                req_ride_dict[trip] = i
        requests['ride_id_' + objective] = pd.Series(req_ride_dict)

        ttrav_sh, u_sh, kinds = dict(), dict(), dict()
        for i, sh in schedules[objective].iterrows():
            for j, trip in enumerate(sh.indexes):
                pos_o = sh['indexes_orig'].index(trip) + 1
                pos_d = sh['indexes_dest'].index(trip) + 1 + len(sh['indexes'])
                ttrav_sh[trip] = sum(sh['times'][pos_o:pos_d])
                u_sh[trip] = sh['u_paxes'][j]
                kinds[trip] = sh['kind']

        requests['ttrav_sh_' + objective] = pd.Series(ttrav_sh)
        requests['u_sh_' + objective] = pd.Series(u_sh)
        requests['kind_' + objective] = pd.Series(kinds)

        requests['position_' + objective] = requests.apply(
            lambda x: schedules[objective].loc[x.ride_id]['indexes'].index(x.name)
            if x.ride_id in schedules[objective].index
            else -1,
            axis=1
        )
        schedules[objective]['degree'] = schedules[objective].apply(lambda x: len(x.indexes), axis=1)

    databank.exmas.recalibrated_rides = rides
    databank.exmas.schedules = {objective: schedules[objective] for objective in objectives}
    databank.exmas.requests = requests

    return databank
