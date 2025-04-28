import numpy as np
import pandas as pd
from dotmap import DotMap
import pulp

from ExMAS.probabilistic_exmas import match, solver_for_pulp


def matching_function(
    databank: DotMap or dict,
    params: DotMap or dict = {},
    objectives: list or None = None,
    min_max: str = "min",
    filter_rides: str or False = False,
    opt_flag: str = "",
    rides_requests: bool = False,
    **kwargs
):
    if objectives is None:
        objectives = databank["exmas"]["objectives"].copy()

    if rides_requests:
        rides = databank['rides']
        requests = databank['requests']
    else:
        rides = databank["exmas"]["recalibrated_rides"]
        requests = databank["exmas"]["requests"]

    if kwargs.get('requestsErrorIndex', False):
        requests.index = requests['index']

    if filter_rides:
        rides.loc[[not t for t in rides[filter_rides]], 'u_veh'] += 10000

    # TODO: this attempt
    if kwargs.get('reindex', False):
        rides['org_index'] = rides['indexes'].copy()
        requests['org_index'] = requests['index'].copy()
        travellers_reindex = {pax_id: new_id for pax_id, new_id in
                              zip(requests['index'], range(len(requests)))}
        rides['temp_index'] = rides['indexes'].apply(
            lambda x: [travellers_reindex[t] for t in x])  # assign new indexes

    schedules = {}

    for objective in objectives:
        params["matching_obj"] = objective
        selected = match(
            im=rides,
            r=requests,
            params=params,
            min_max=min_max
        )
        if filter_rides:
            rides.loc[[not t for t in rides[filter_rides]], 'u_veh'] -= 10000

        rides["selected_" + objective + opt_flag] = pd.Series(selected.copy())

        schedules[objective + opt_flag] = rides.loc[[bool(t) for t in rides["selected_" + objective + opt_flag]]].copy()

        req_ride_dict = dict()
        for i, trips in schedules[objective + opt_flag].iterrows():
            for trip in trips.indexes:
                req_ride_dict[trip] = i
        requests['ride_id_' + objective] = pd.Series(req_ride_dict)

        ttrav_sh, u_sh, kinds = dict(), dict(), dict()
        for i, sh in schedules[objective + opt_flag].iterrows():
            for j, trip in enumerate(sh.indexes):
                try:
                    pos_o = sh['indexes_orig'].index(trip) + 1
                    pos_d = sh['indexes_dest'].index(trip) + 1 + len(sh['indexes'])
                    ttrav_sh[trip] = sum(sh['times'][pos_o:pos_d])
                except TypeError:
                    pos_o = sh['indexes_orig'].index(str(trip)) + 1
                    pos_d = sh['indexes_dest'].index(str(trip)) + 1 + len(sh['indexes'])
                    ttrav_sh[trip] = sum(eval(sh['times'])[pos_o:pos_d])
                u_sh[trip] = sh['u_paxes'][j]
                kinds[trip] = sh['kind']

        requests['ttrav_sh_' + objective + opt_flag] = pd.Series(ttrav_sh)
        requests['u_sh_' + objective + opt_flag] = pd.Series(u_sh)
        requests['kind_' + objective + opt_flag] = pd.Series(kinds)

        if kwargs.get('requestsErrorIndex', False):
            pass
        else:
            requests['position_' + objective + opt_flag] = requests.apply(
                lambda x: schedules[objective + opt_flag].loc[x.ride_id]['indexes'].index(x.name)
                if x.ride_id in schedules[objective + opt_flag].index
                else -1,
                axis=1
            )
        schedules[objective + opt_flag]['degree'] = schedules[objective + opt_flag].apply(lambda x: len(x.indexes), axis=1)

    if not rides_requests:
        databank['exmas']['recalibrated_rides'] = rides
        if len(databank['exmas'].get('schedules', [])) > 0:
            for objective in objectives:
                databank['exmas']['schedules'][objective + opt_flag] = schedules[objective + opt_flag]
        else:
            databank['exmas']['schedules'] = {objective + opt_flag: schedules[objective + opt_flag]
                                              for objective in objectives}
        databank['exmas']['requests'] = requests

        return databank
    else:
        return {'rides': rides,
                'requests': requests,
                'schedules': schedules}


def matching_function_light(
        _rides: pd.DataFrame,
        _requests: pd.DataFrame,
        _objective: str = "objective",
        _min_max: str = "max",
        **kwargs
) -> pd.DataFrame or dict:
    # _rides['PassHourTrav_ns'] = _rides.apply(lambda x: sum([_requests.loc[_].ttrav for _ in x.indexes]), axis=1)

    if _min_max == "min":
        prob = pulp.LpProblem("Matching problem", pulp.LpMinimize)  # problem
    else:
        prob = pulp.LpProblem("Matching problem", pulp.LpMaximize)

    variables = pulp.LpVariable.dicts("r", range(len(_rides)), cat='Binary')  # decision variables
    costs = _rides[_objective] # cost
    prob += pulp.lpSum([variables[i] * costs[i] for i in variables]), 'ObjectiveFun' # add to the problem

    # constrains
    travellers_reindex = {pax_id: new_id for pax_id, new_id in
                          zip(_requests['index'], range(len(_requests)))}
    _rides['temp_index'] = _rides['indexes'].apply(lambda x: [travellers_reindex[t] for t in x]) # assign new indexes

    def _binary_row(_row, _total_len):
        out = np.zeros(_total_len)
        for new_id in _row['temp_index']:
            out[new_id] = 1
        return out

    constraint_array = np.vstack(
        _rides.apply(_binary_row, _total_len=len(travellers_reindex.keys()), axis=1)
    ).T

    for _num, _travellers_ride in enumerate(constraint_array):
        prob += pulp.lpSum([_travellers_ride[i] * variables[i] for i in variables
                            if _travellers_ride[i] > 0]) == 1, 'c' + str(_num)

    # back to the problem
    solver = pulp.getSolver(solver_for_pulp())
    solver.msg = False
    prob.solve(solver)

    _rides = _rides.drop(columns=['temp_index'])

    _selected = [0]*constraint_array.shape[1]
    for variable in prob.variables():
        _selected[int(str(variable)[2:])] = variable.varValue

    if kwargs.get('rrs_output', False):
        return {'rides': _rides,
                'requests': _requests,
                'schedules': {'objective': _rides.loc[[bool(t) for t in _selected]]}}

    return _rides.loc[[bool(t) for t in _selected]]

