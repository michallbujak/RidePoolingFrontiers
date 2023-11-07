import pandas as pd
from dotmap import DotMap


def evaluate_pooling(
        databank: DotMap or dict
) -> DotMap or dict:
    databank['exmas']['results'] = {}

    for objective in databank['exmas']['schedules'].keys():
        db_copy = databank.copy()
        _results = {}

        _schedule = db_copy['exmas']['schedules'][objective]
        _requests = db_copy['exmas']['requests']

        _schedule['ttrav'] = _schedule.apply(lambda x: sum(x.times[1:]), axis=1)

        _results['VehHourTrav'] = _schedule.ttrav.sum()
        _results['VehHourTrav_ns'] = _requests.ttrav.sum()

        _results['PassHourTrav'] = _requests.ttrav_sh.sum()
        _results['PassHourTrav_ns'] = _requests.ttrav.sum()

        _results['PassUtility'] = _requests.u_sh.sum()
        _results['PassUtility_ns'] = _requests.u.sum()

        _results[objective] = _schedule[objective].sum()

        _results['nR'] = _requests.shape[0]
        _results['SINGLE'] = _schedule[(_schedule.kind == 1)].shape[0]
        _results['PAIRS'] = _schedule[(_schedule.kind > 1) & (_schedule.kind < 30)].shape[0]
        _results['TRIPLES'] = _schedule[(_schedule.kind >= 30) & (_schedule.kind < 40)].shape[0]
        _results['QUADRIPLES'] = _schedule[(_schedule.kind >= 40) & (_schedule.kind < 50)].shape[0]
        _results['QUINTETS'] = _schedule[(_schedule.kind >= 50) & (_schedule.kind < 100)].shape[0]
        _results['PLUS5'] = _schedule[(_schedule.kind == 100)].shape[0]
        _results['shared_ratio'] = 1 - _results['SINGLE'] / _results['nR']

        # assign to the variable
        databank['exmas']['results'][objective] = _results.copy()

    return databank
