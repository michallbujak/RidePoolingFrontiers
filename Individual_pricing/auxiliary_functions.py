import pandas as pd
import numpy as np
from dotmap import DotMap


def evaluate_pooling(
        databank: DotMap or dict,
        deterministic: bool = True
) -> DotMap or dict:
    databank['exmas']['results'] = {}

    for objective in list(databank['exmas']['schedules'].keys()) + ["default"]:
        db_copy = databank.copy()
        _results = {}

        if objective == "default":
            _schedule = db_copy['exmas']['recalibrated_rides']
            _schedule = _schedule.loc[_schedule["selected"] == 1]
        else:
            _schedule = db_copy['exmas']['schedules'][objective]

        _requests = db_copy['exmas']['requests']

        _schedule['ttrav'] = _schedule.apply(lambda x: sum(x.times[1:]), axis=1)

        _results['VehHourTrav'] = _schedule.ttrav.sum()
        _results['VehHourTrav_ns'] = _requests.ttrav.sum()

        if objective != "default":
            _results['PassHourTrav'] = _requests['ttrav_sh_' + objective].sum()
        else:
            _results['PassHourTrav'] = _requests['ttrav_sh'].sum()

        _results['PassHourTrav_ns'] = _requests.ttrav.sum()

        _results['PassUtility'] = _requests.u_sh.sum()
        _results['PassUtility_ns'] = _requests.u.sum()

        _results['nR'] = _requests.shape[0]
        _results['SINGLE'] = _schedule[(_schedule.kind == 1)].shape[0]
        _results['PAIRS'] = _schedule[(_schedule.kind > 1) & (_schedule.kind < 30)].shape[0]
        _results['TRIPLES'] = _schedule[(_schedule.kind >= 30) & (_schedule.kind < 40)].shape[0]
        _results['QUADRIPLES'] = _schedule[(_schedule.kind >= 40) & (_schedule.kind < 50)].shape[0]
        _results['QUINTETS'] = _schedule[(_schedule.kind >= 50) & (_schedule.kind < 100)].shape[0]
        _results['PLUS5'] = _schedule[(_schedule.kind == 100)].shape[0]
        _results['shared_ratio'] = 1 - _results['SINGLE'] / _results['nR']

        if deterministic:
            if objective[-3:] == "max":
                _schedule["profit"] = _schedule["profit_max"]
                _schedule["profitability"] = _schedule["profitability_max"]
            else:
                _schedule["profit"] = _schedule["profit_base"]
                _schedule["profitability"] = _schedule["profitability_base"]

            _results["Profit"] = _schedule["profit"].sum()
            _results["Average_profitability"] = np.mean(_schedule["profitability"] / 1000)

        else:
            pass

        # assign to the variable
        databank['exmas']['results'][objective] = _results.copy()

    return databank


def compare_objective_methods(
        databank: DotMap or dict
) -> DotMap or dict:
    all_results = {
        "u_veh": databank["exmas"]["res"]
    }
    for _key, _res in databank["exmas"]["results"].items():
        all_results[_key] = _res

    return pd.DataFrame(all_results)


def aggregate_results(
        results: list
) -> pd.DataFrame:
    pass


def extract_selected_profitability(
        _rides: pd.DataFrame,
        _disc_names: list
):
    out = {
        'Personalised': _rides.loc[_rides['selected_profitability'] == 1]
    }
    for _disc in _disc_names:
        out[_disc] = _rides.loc[_rides['selected_' + _disc + '_profitability'] == 1]
    return out


def extract_selected_discounts(
        _rides: pd.DataFrame,
        _disc_names: list
):
    tmp_dat = _rides.loc[_rides['selected_profitability'] == 1]

    out_list = []
    out = {
        'Personalised': list(tmp_dat['best_profit'].apply(lambda x: x[5]))
    }
    out_list += [list(tmp_dat['best_profit'].apply(lambda x: x[5]))]

    for _disc in _disc_names:
        tmp_dat = _rides.loc[_rides['selected_' + _disc + '_profitability'] == 1]
        td = tmp_dat.apply(
            lambda x: x[_disc + '_profitability'] / len(x['indexes']),
            axis=1
        )
        out['Flat disc. 0.' + _disc[1:]] = list(td)
        out_list += [list(td)]

    return out, out_list

def bracket(ax, pos=[0, 0], scalex=1, scaley=1, text="", textkw=None, linekw=None):
    if textkw is None:
        textkw = dict()
    if linekw is None:
        linekw = dict()
    x = np.array([0, 0.05, 0.45, 0.5])
    y = np.array([0, -0.01, -0.01, -0.02])
    x = np.concatenate((x, x + 0.5))
    y = np.concatenate((y, y[::-1]))
    ax.plot(x * scalex + pos[0], y * scaley + pos[1], clip_on=False,
            transform=ax.get_xaxis_transform(), **linekw)
    ax.text(pos[0] + 0.5 * scalex, (y.min() - 0.01) * scaley + pos[1], text,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", **textkw)
