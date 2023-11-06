import pandas as pd
from dotmap import DotMap

from ExMAS.probabilistic_exmas import match

def matching(
    databank: DotMap,
    params: DotMap,
    objectives: list or None = None
):
    if objectives is not None:
        params.matching_obj = objectives
    else:
        params.matching_obj = [params.matching_obj]

    rides = databank["exmas"]["rides"]
    requests = databank["exmas"]["requests"]

    for objective in params.matching_obj:
        rides["selected_" + objective] = match(
            im=rides,
            r=requests,
            params=params
        )

    schedule = rides[selected].copy()

    req_ride_dict = dict()
    for i, trips in schedule.iterrows():
        for trip in trips.indexes:
            req_ride_dict[trip] = i
    requests['ride_id'] = pd.Series(req_ride_dict)
    ttrav_sh, u_sh, kinds = dict(), dict(), dict()
    for i, sh in schedule.iterrows():
        for j, trip in enumerate(sh.indexes):
            pos_o = sh.indexes_orig.index(trip) + 1
            pos_d = sh.indexes_dest.index(trip) + 1 + len(sh.indexes)
            ttrav_sh[trip] = sum(sh.times[pos_o:pos_d])
            u_sh[trip] = sh.u_paxes[j]
            kinds[trip] = sh.kind

    requests['ttrav_sh'] = pd.Series(ttrav_sh)
    requests['u_sh'] = pd.Series(u_sh)
    requests['kind'] = pd.Series(kinds)

    requests['position'] = requests.apply(
        lambda x: schedule.loc[x.ride_id].indexes.index(x.name) if x.ride_id in schedule.index else -1, axis=1)
    schedule['degree'] = schedule.apply(lambda x: len(x.indexes), axis=1)

    if make_assertion:  # test consitency
        assert opt_outs or len(
            requests.ride_id) - requests.ride_id.count() == 0  # all trips are assigned
        to_assert = requests[requests.ride_id >= 0]  # only non optouts

        assert (to_assert.u_sh <= to_assert.u + 0.5).all
        assert (to_assert.ttrav <= (to_assert.ttrav_sh + 3)).all()
        if multi_platform_matching:
            for i in schedule.index.values:
                # check if all schedules are for travellers from the same platform
                assert _inData.requests.loc[schedule.loc[i].indexes].platform.nunique() == 1

    # store the results back
    _inData.exmas.rides = rides
    _inData.exmas.schedule = schedule
    _inData.exmas.requests = requests
    return _inData