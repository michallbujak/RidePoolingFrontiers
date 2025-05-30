"""
# Revised ExMAS algorithm
> Exact Matching of Attractive Shared rides (ExMAS) for system-wide strategic evaluations
---

![MAP](/data/map.png)

ExMAS allows you to match trips into attractive shared rides.

For a given:
* network (`osmnx` graph),
* demand (microscopic set of trips $q_i = (o_i, d_i, t_i)$)
* parameters (behavioural, like _willingness-to-share_ and system like _discount_ for shared rides)

It computes:
* optimal set of shared rides (results of bipartite matching with a given objective)
* shareability graph
* set of all feasible rides
* KPIs of sharing
* trip sharing attributes

ExMAS is a `python` based open-source package applicable to general networks and demand patterns.

If you find this code useful in your research, please consider citing:

>_Kucharski R. , Cats. O 2020.
Exact matching of attractive shared rides (ExMAS) for system-wide strategic evaluations,
Transportation Research Part B 139 (2020) 285-310_
https://doi.org/10.1016/j.trb.2020.06.006


[Quickstart tutorial](https://github.com/RafalKucharskiPK/ExMAS/blob/master/notebooks/ExMAS.ipynb)

----
Original version:
Rafał Kucharski, TU Delft, 2020 r.m.kucharski (at) tudelft.nl

Probabilistic version (current/below):
Michał Bujak, JU Kraków, 2024 michal.bujak (at) doctoral.uj.edu.pl
"""

__author__ = "Rafal Kucharski"
__copyright__ = "Copyright 2020, TU Delft"
__credits__ = ["Oded Cats, Arjan de Ruijter, Subodh Dubey, Nejc Gerzinic"]
__version__ = "1.0.1"
__maintainer__ = "Michal Bujak"
__email__ = "michal.bujak _at_ doctoral.uj.edu.pl"

import ast
import sys
from itertools import product
import logging
from types import FunctionType
import warnings

from dotmap import DotMap
from enum import Enum

import numpy as np
import pandas as pd
import networkx as nx
import pulp
import platform

import matplotlib.pyplot as plt

from ExMAS.utils import mixed_discrete_norm_distribution

pd.options.mode.chained_assignment = None

try:
    np.warnings.filterwarnings('ignore')
except AttributeError:
    warnings.filterwarnings('ignore')

##########
# CONSTS #
##########

# columns of ride-candidates DataFrame
RIDE_COLS = ['indexes', 'u_pax', 'u_veh', 'kind', 'u_paxes', 'times', 'indexes_orig', 'indexes_dest', 'true_u_pax',
             'true_u_paxes', 'delays']


class SbltType(Enum):  # type of shared ride. first digit is the degree, second is type (FIFO/LIFO/other)
    SINGLE = 1
    FIFO2 = 20
    LIFO2 = 21
    TRIPLE = 30
    FIFO3 = 30
    LIFO3 = 31
    MIXED3 = 32
    FIFO4 = 40
    LIFO4 = 41
    MIXED4 = 42
    FIFO5 = 50
    LIFO5 = 51
    MIXED5 = 52
    PLUS5 = 100


##############
# MAIN CALLS #
##############

# ALGORITHM 3
def main(input_data, params, plot=False):
    """
    main call
    :param input_data: input (graph, requests, .. )
    :param exmas_params: parameters
    :param plot: flag to plot charts for consecutive steps
    :return: inData.exmas.schedule - selecgted shared rides
    inData.exmas.rides - all ride candidates
    inData.exmas.res -  KPIs
    @param _seed:
    """
    _inData = input_data.copy()
    if type(_inData) == dict:
        _inData = DotMap(_inData)

    _inData.logger = init_log(params)  # initialize console logger

    _inData = sample_random_parameters(_inData, params)
    _inData = add_noise(_inData, params)

    check_if_correct_attributes(params)

    _inData = single_rides(_inData, params)  # prepare requests as a potential single rides
    degree = 1

    _inData = pairs(_inData, params, plot=plot)
    degree = 2

    _inData.logger.info('Degree {} \tCompleted'.format(degree))

    if degree < params.max_degree:
        _inData = make_shareability_graph(_inData, params)

        while degree < params.max_degree and _inData.exmas.R[degree].shape[0] > 0:
            _inData.logger.info('trips to extend at degree {} : {}'.format(degree,
                                                                           _inData.exmas.R[degree].shape[0]))
            _inData = extend_degree(_inData, params, degree)
            degree += 1
            _inData.logger.info('Degree {} \tCompleted'.format(degree))
        if degree == params.max_degree:
            _inData.logger.info('Max degree reached {}'.format(degree))
            _inData.logger.info('Trips still possible to extend at degree {} : {}'.format(degree,
                                                                                          _inData.exmas.R[degree].shape[
                                                                                              0]))
        else:
            _inData.logger.info(('No more trips to exted at degree {}'.format(degree)))

    _inData.exmas.rides = _inData.exmas.rides.reset_index(drop=True)  # set index
    _inData.exmas.rides['index'] = _inData.exmas.rides.index  # copy index

    if params.get('without_matching', False):
        return _inData  # quit before matching
    else:
        _inData = matching(_inData, params, plot=plot)
        _inData.logger.info('Calculations  completed')
        _inData = evaluate_shareability(_inData, params, plot=plot)

        return _inData


########
# CORE #
########


def single_rides(_inData, params):
    """
    prepare _inData.requests for calculations
    :param _inData:
    :param params: parameters
    :return:
    """

    def f_delta():
        # maximal possible delay of a trip (computed before join)
        return list(map(lambda x: x if x >= 0 else 0, (1 / req.WtS - 1) * req.ttrav + \
                        (params.price * params.shared_discount * req.dist / 1000) / (req.VoT * req.WtS)))

    # prepare data structures
    _inData.exmas = DotMap(_dynamic=False)
    _inData.exmas.log = DotMap(_dynamic=False)
    _inData.exmas.log.sizes = DotMap(_dynamic=False)
    # prepare requests
    req = _inData.requests.copy().sort_index()
    if params.get('reset_ttrav', True):
        # reset times, reindex
        t0 = req.treq.min()  # set 0 as the earliest departure time
        req.treq = (req.treq - t0).dt.total_seconds().astype(int)  # recalc times for seconds starting from zero
        req.ttrav = req.ttrav.dt.total_seconds().divide(params.avg_speed).astype(int)  # recalc travel times using speed

    """ NEW """
    if "VoT" in _inData.prob.sampled_random_parameters.columns:
        req = pd.merge(req, _inData.prob.sampled_random_parameters['VoT'], left_index=True, right_index=True)
    else:
        req["VoT"] = params.VoT

    if "WtS" in _inData.prob.sampled_random_parameters.columns:
        req = pd.merge(req, _inData.prob.sampled_random_parameters['WtS'], left_index=True, right_index=True)
    else:
        req["WtS"] = params.WtS

    if "delay_value" in _inData.prob.sampled_random_parameters.columns:
        req = pd.merge(req, _inData.prob.sampled_random_parameters['delay_value'], left_index=True, right_index=True)
    else:
        req["delay_value"] = params.delay_value
    """ END OF NEW"""

    req['delta'] = f_delta()  # assign maximal delay in seconds
    req['true_u'] = params.price * req.dist / 1000 + req.VoT * req.ttrav
    if params.get("noise", None) is not None:
        assert isinstance(params.noise, dict), "Incorrect type of exmas_params.noise in json (expected dict)"
        req['u'] = req["true_u"] + np.random.normal(size=len(req), loc=params.noise.get("mean", 0),
                                                    scale=params.noise.get("st_dev", 0))
    else:
        req['u'] = req["true_u"]

    req['u'] = req['u'].apply(lambda x: x if x >= 0 else 0)

    req = req.sort_values(['treq', 'pax_id'])  # sort
    req = req.reset_index()

    try:
        _inData.prob.sampled_random_parameters
    except NameError:
        var_exists = False
    else:
        var_exists = True

    if var_exists:
        req["new_index"] = req.index
        _inData.prob.sampled_random_parameters = \
            pd.merge(_inData.prob.sampled_random_parameters, req[["id", "new_index"]], left_index=True, right_on="id")
        _inData.prob.sampled_random_parameters.set_index("id", drop=True, inplace=True)
        req.drop(columns="new_index", inplace=True)

    # output
    _inData.exmas.requests = req.copy()
    df = req.copy()
    df['kind'] = SbltType.SINGLE.value  # assign a type for a ride
    df['indexes'] = df.index
    df['times'] = df.apply(lambda x: [x.treq, x.ttrav], axis=1)  # sequence of travel times
    df = df[['indexes', 'u', 'ttrav', 'kind', 'times', 'true_u']]  # columns to store as a shared ride
    df['indexes'] = df['indexes'].apply(lambda x: [x])
    df['u_paxes'] = df['u'].apply(lambda x: [x])
    df['true_u_paxes'] = df['true_u'].apply(lambda x: [x])
    df['delays'] = [[0] for j in range(len(df))]

    df.columns = ['indexes', 'u_pax', 'u_veh', 'kind', 'times', 'true_u_pax', 'u_paxes', 'true_u_paxes', 'delays']  # synthax for the output rides
    # df = df[['indexes', 'u_pax', 'u_veh', 'kind', 'u_paxes', 'times', 'true_u_pax', 'true_u_paxes', 'delays']]
    df['indexes_orig'] = df.indexes  # copy order of origins for single rides
    df['indexes_dest'] = df.indexes  # and dest
    df = df[RIDE_COLS]

    _inData.exmas.SINGLES = df.copy()  # single trips
    _inData.exmas.log.sizes[1] = {'potential': df.shape[0], 'feasible': df.shape[0]}
    _inData.exmas.rides = df.copy()

    _inData.exmas.R = dict()  # all the feasible rides
    _inData.exmas.R[1] = df.copy()  # of a given degree

    return _inData


# ALGORITHM 1
def pairs(_inData, params, process=True, check=True, plot=False):
    """
    Identifies pair-wise shareable trips S_ij, i.e. for which utility of shared ride is greater than utility of
    non-shared ride for both trips i and j.
    First S_ij.FIFO2 trips are identified, i.e. sequence o_i,o_j,d_i,d_j.
    Subsequently, from FIFO2 trips we identify LIFO2 trips, i.e. o_i,o_j,d_j,d_i

    :param _inData: main data structure, with .skim (node x node dist matrix) , .requests (with origin, dest and treq)
    :param params: .json populated dictionary of parameters
    :param process: boolean flag to calculate measures at the end of calulations
    :param check: run test to make sure results are consistent
    :param plot: plot matrices illustrating the shareability
    :return: _inData with .exmas
    """
    # input
    req = _inData.exmas.requests.copy()  # work with single requests

    # VECTORIZED FUNCTIONS TO QUICKLY COMPUTE FORMULAS ALONG THE DATAFRAME
    def utility_ns_i():
        # utility of non-shared trip i
        return params.price * r.dist_i / 1000 + r.VoT_i * r.ttrav_i

    def utility_ns_j():
        # utility of non shared trip j
        return params.price * r.dist_j / 1000 + r.VoT_j * r.ttrav_j

    def utility_sh_i():
        # utility of shared trip i
        return (params.price * (1 - params.shared_discount) * r.dist_i / 1000 +
                r.VoT_i * r.WtS_i * (r.t_oo + params.pax_delay + r.t_od + r.delay_value_i * abs(r.delay_i)))

    def utility_sh_j():
        # utility of shared trip j
        return (params.price * (1 - params.shared_discount) * r.dist_j / 1000 +
                r.VoT_j * r.WtS_j * (r.t_od + r.t_dd + params.pax_delay +
                                     r.delay_value_j * abs(r.delay_j)))

    def utility_i():
        # difference u_sh_i - u_ns_i (has to be positive)
        return (params.price * r.dist_i / 1000 * params.shared_discount
                + r.VoT_i * (r.ttrav_i - r.WtS_i * (
                        r.t_oo + r.t_od + params.pax_delay + r.delay_value_i * abs(r.delay_i))))

    def utility_j():
        # difference u_sh_i - u_ns_i
        return (params.price * r.dist_j / 1000 * params.shared_discount
                + r.VoT_j * (r.ttrav_j - r.WtS_j * (
                        r.t_od + r.t_dd + params.pax_delay + r.delay_value_j * abs(r.delay_j))))

    def utility_i_LIFO():
        # utility of LIFO trip for i
        return params.price * r.dist_i / 1000 * params.shared_discount + r.VoT_i * (r.ttrav_i - r.WtS_i * (
                r.t_oo + r.t_od + 2 * params.pax_delay + r.t_dd + params.delay_value * abs(r.delay_i)))

    def utility_j_LIFO():
        # utility of LIFO trip for j
        return params.price * r.dist_j / 1000 * params.shared_discount + \
               r.VoT_j * (r.ttrav_j - r.WtS_j * (r.t_od + params.delay_value * abs(r.delay_j)))

    def utility_sh_i_LIFO():
        # difference u_sh_i_LIFO - u_ns_i
        return (params.price * (1 - params.shared_discount) * r.dist_i / 1000 +
                r.VoT_i * r.WtS_i * (
                        r.t_oo + r.t_od + r.t_dd + 2 * params.pax_delay + r.delay_value_i * abs(r.delay_i)))

    def utility_sh_j_LIFO():
        # difference u_sh_j_LIFO - u_ns_j
        return (params.price * (1 - params.shared_discount) * r.dist_j / 1000 +
                r.VoT_j * r.WtS_j * (r.t_od + r.delay_value_j * abs(r.delay_j)))

    def query_skim(r, _from, _to, _col, _filter=True):
        """
        returns trip times for given node pair _from, _to and stroes into _col of df
        :param r: current set of queries
        :param _from: column name in r designating origin
        :param _to: column name in r designating destination
        :param _col: name of column in 'r' where matrix entries are stored
        :param _filter: do we filter the skim for faster queries (used always apart from the last query for LIFO2)
        :return: attributes in r
        """
        #
        if _filter:
            skim = the_skim.loc[
                r[_from].unique(), r[_to].unique()].unstack().to_frame()  # reduce the skim size for faster join
        else:
            skim = the_skim.unstack().to_frame()  # unstack and to_frame for faster column representation of matrix
        # skim matrix is unstacked (column vector) with two indexes
        skim.index.names = ['o', 'd']  # unify names for join
        skim.index = skim.index.set_names("o", level=0)
        skim.index = skim.index.set_names("d", level=1)

        skim.columns = [_col]
        # requests now has also two indexes
        r = r.set_index([_to, _from], drop=False)
        r.index = r.index.set_names("o", level=0)
        r.index = r.index.set_names("d", level=1)

        return r.join(skim, how='left')  # perform the jin to get the travel time for each request

    def sp_plot(_r, r, nCall, title):
        # function to plot binary shareability matrix at respective stages
        _r[1] = 0  # init boolean column
        if nCall == 0:
            _r.loc[r.index, 1] = 1  # initialize for first call
            sizes['initial'] = params.nP * params.nP
        else:
            _r.loc[r.set_index(['i', 'j']).index, 1] = 1
        sizes[title] = r.shape[0]
        _inData.logger.info(str(r.shape[0]) + '\t' + title)
        mtx = _r[1].unstack().values
        axes[nCall].spy(mtx)
        axes[nCall].set_title(title)
        axes[nCall].set_xticks([])

    if plot:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    _r = None

    sizes = dict()
    # MAIN CALULATIONS
    _inData.logger.info('Initializing pairwise trip shareability between {0} and {0} trips.'.format(params.nP))
    r = pd.DataFrame(index=pd.MultiIndex.from_product([req.index, req.index]))  # new df with a pairwise index
    _inData.logger.info('creating combinations')
    cols = ['origin', 'destination', 'ttrav', 'treq', 'delta', 'dist', 'VoT', 'WtS', "delay_value"]
    r[[col + "_i" for col in cols]] = req.loc[r.index.get_level_values(0)][cols].set_index(r.index)  # assign columns
    r[[col + "_j" for col in cols]] = req.loc[r.index.get_level_values(1)][cols].set_index(r.index)  # time consuming

    r['i'] = r.index.get_level_values(0)  # assign index to columns
    r['j'] = r.index.get_level_values(1)
    r = r[~(r.i == r.j)]  # remove diagonal
    _inData.logger.info(str(r.shape[0]) + '\t nR*(nR-1)')
    _inData.exmas.log.sizes[2] = {'potential': r.shape[0]}

    # first condition (before querying any skim)
    if params.horizon > 0:
        r = r[abs(r.treq_i - r.treq_j) < params.horizon]
    q = '(treq_j + delta_j >= treq_i - delta_i)  & (treq_j - delta_j <= (treq_i + ttrav_i + delta_i))'
    r = r.query(q)  # this reduces the size of matrix quite a lot
    if plot:
        _r = r.copy()
        _r[1] = 0
        sp_plot(_r, r, 0, 'departure compatibility')
    if len(r) == 0:
        _inData.exmas.FIFO2 = r  # early exit with empty result
        return _inData

    # make the skim smaller  (query only between origins and destinations)
    skim_indices = list(set(r.origin_i.unique()).union(r.origin_j.unique()).union(
        r.destination_j.unique()).union(r.destination_j.unique()))  # limit skim to origins and destination only
    the_skim = _inData.skim.loc[skim_indices, skim_indices].div(params.avg_speed).astype(int)
    _inData.the_skim = the_skim

    r = query_skim(r, 'origin_i', 'origin_j', 't_oo')  # add t_oo to main dataframe (r)
    q = '(treq_i + t_oo + delta_i >= treq_j - delta_j) & (treq_i + t_oo - delta_i <= treq_j + delta_j)'
    r = r.query(q)  # can we arrive at origin of j within his time window?

    # now we can see if j is reachebale from i with the delay acceptable for both
    # determine delay for i and for j (by def delay/2, otherwise use bound of one delta and remainder for other trip)
    r['delay'] = r.treq_i + r.t_oo - r.treq_j
    # TODO
    r['delay_i'] = r.apply(lambda x: min(abs(x.delay / 2), x.delta_i, x.delta_j) * (1 if x.delay < 0 else -1), axis=1)
    r['delay_j'] = r.delay + r.delay_i

    r = r[abs(r.delay_j) <= r.delta_j / params.delay_value]  # filter for acceptable
    r = r[abs(r.delay_i) <= r.delta_i / params.delay_value]
    if plot:
        sp_plot(_r, r, 1, 'origins shareability')
    if len(r) == 0:
        _inData.exmas.FIFO2 = r  # early exit with empty result
        return _inData

    r = query_skim(r, 'origin_j', 'destination_i', 't_od')

    r['true_utility_i'] = list(utility_i())

    sampled_noise = dict()

    if params.get("panel_noise", None) is not None:
        assert isinstance(params.panel_noise, dict), "Incorrect type of panel_noise in json (should be dict)"
        r = r.merge(pd.DataFrame(_inData.prob.panel_noise, columns=['panel_noise_i']), left_on='i', right_index=True)
        r = r.merge(pd.DataFrame(_inData.prob.panel_noise, columns=['panel_noise_j']), left_on='j', right_index=True)

    if params.get("noise", None) is not None:
        assert isinstance(params.noise, dict), "Incorrect type of noise in json (should be dict)"

    r, sampled_noise = utility_for_r(r, "i", params, sampled_noise, 1)

    if plot:
        sp_plot(_r, r, 2, 'utility for i')
    if len(r) == 0:
        _inData.exmas.FIFO2 = r
        return _inData
    rLIFO = r.copy()

    r = query_skim(r, 'destination_i', 'destination_j', 't_dd')  # and now see if it is attractive also for j
    # now we have times for all segments: # t_oo_i_j # t_od_j_i # dd_i_j
    # let's compute utility for j
    # r = r[utility_j() > 0]

    r['true_utility_j'] = list(utility_j())

    r, sampled_noise = utility_for_r(r, "j", params, sampled_noise, 1)

    if plot:
        sp_plot(_r, r, 3, 'utility for j')
    if len(r) == 0:
        _inData.exmas.FIFO2 = r
        return _inData

    # profitability
    r['ttrav'] = r.t_oo + r.t_od + r.t_dd + 2 * params.pax_delay

    r = r.set_index(['i', 'j'], drop=False)  # done - final result of pair wise FIFO shareability

    if process:
        # lets compute some more measures
        r['kind'] = SbltType.FIFO2.value
        r['indexes'] = r.apply(lambda x: [int(x.i), int(x.j)], axis=1)
        r['indexes_orig'] = r.indexes
        r['indexes_dest'] = r.apply(lambda x: [int(x.i), int(x.j)], axis=1)

        r['true_u_i'] = utility_sh_i().copy()
        r['true_u_j'] = utility_sh_j().copy()

        r = calculate_r_utility(r, "i", params, sampled_noise, 1)
        r = calculate_r_utility(r, "j", params, sampled_noise, 1)

        r['t_i'] = r.t_oo + r.t_od + params.pax_delay
        r['t_j'] = r.t_od + r.t_dd + params.pax_delay
        r['delta_ij'] = r.apply(lambda x: x.delta_i - params.delay_value * abs(x.delay_i) - (x.t_i - x.ttrav_i), axis=1)
        r['delta_ji'] = r.apply(lambda x: x.delta_j - params.delay_value * abs(x.delay_j) - (x.t_j - x.ttrav_j), axis=1)
        r['delta'] = r[['delta_ji', 'delta_ij']].min(axis=1)
        r['u_pax'] = r['u_i'] + r['u_j']
        r['true_u_pax'] = r['true_u_i'] + r['true_u_j']

    _inData.exmas.FIFO2 = r.copy()
    del r

    # LIFO2
    r = rLIFO
    r = query_skim(r, 'destination_j', 'destination_i', 't_dd')  # set different sequence of times
    r.t_od = r.ttrav_j

    r["true_utility_i"] = list(utility_i_LIFO())

    r, sampled_noise = utility_for_r(r, "i", params, sampled_noise, 2)

    r["true_utility_j"] = list(utility_j_LIFO())

    r, sampled_noise = utility_for_r(r, "j", params, sampled_noise, 2)

    r = r.set_index(['i', 'j'], drop=False)
    r['ttrav'] = r.t_oo + r.t_od + r.t_dd + 2 * params.pax_delay

    if plot:
        _inData.logger.info(str(r.shape[0]) + '\tLIFO pairs')
        sizes['LIFO'] = r.shape[0]

    if r.shape[0] > 0 and process:
        r['kind'] = SbltType.LIFO2.value

        r['indexes'] = r.apply(lambda x: [int(x.i), int(x.j)], axis=1)
        r['indexes_orig'] = r.indexes
        r['indexes_dest'] = r.apply(lambda x: [int(x.j), int(x.i)], axis=1)

        r['true_u_i'] = utility_sh_i_LIFO()
        r['true_u_j'] = utility_sh_j_LIFO()

        r = calculate_r_utility(r, "i", params, sampled_noise, 2)
        r = calculate_r_utility(r, "j", params, sampled_noise, 2)

        # r['u_i'] = r['true_u_i'] + r['panel_noise_i']
        # r['u_j'] = r['true_u_j'] + r['panel_noise_j']

        r['t_i'] = r.t_oo + r.t_od + r.t_dd + 2 * params.pax_delay
        r['t_j'] = r.t_od
        r['delta_ij'] = r.apply(
            lambda x: x.delta_i - params.delay_value * abs(x.delay_i) - (x.t_oo + x.t_od + x.t_dd - x.ttrav_i), axis=1)
        r['delta_ji'] = r.apply(lambda x: x.delta_j - params.delay_value * abs(x.delay_j), axis=1)
        r['delta'] = r[['delta_ji', 'delta_ij']].min(axis=1)
        r['u_pax'] = r['u_i'] + r['u_j']

        r['true_u_pax'] = r['true_u_i'] + r['true_u_j']

    _inData.exmas.LIFO2 = r.copy()
    _inData.exmas.pairs = pd.concat([_inData.exmas.FIFO2, _inData.exmas.LIFO2],
                                    sort=False).set_index(['i', 'j', 'kind'],
                                                          drop=False).sort_index()
    _inData.exmas.log.sizes[2]['feasible'] = _inData.exmas.LIFO2.shape[0] + _inData.exmas.FIFO2.shape[0]
    _inData.exmas.log.sizes[2]['feasibleFIFO'] = _inData.exmas.FIFO2.shape[0]
    _inData.exmas.log.sizes[2]['feasibleLIFO'] = _inData.exmas.LIFO2.shape[0]

    for df in [_inData.exmas.FIFO2.copy(), _inData.exmas.LIFO2.copy()]:
        if df.shape[0] > 0:
            df['u_paxes'] = df.apply(lambda x: [x.u_i, x.u_j], axis=1)
            df['true_u_paxes'] = df.apply(lambda x: [x.true_u_i, x.true_u_j], axis=1)
            df['u_veh'] = df.ttrav
            df['times'] = df.apply(
                lambda x: [x.treq_i + x.delay_i, x.t_oo + params.pax_delay, x.t_od, x.t_dd], axis=1)

            df['delays'] = [[abs(a), abs(b)] for a, b in zip(df['delay_i'], df['delay_j'])]

            df = df[RIDE_COLS]

            _inData.exmas.rides = pd.concat([_inData.exmas.rides, df], sort=False)
    gain = (1 - float(r.shape[0]) / (params.nP * (params.nP - 1))) * 100
    _inData.logger.info('Reduction of feasible pairs by {:.2f}%'.format(gain))
    if plot:
        if 'figname' in params.keys():
            plt.savefig(params.figname)
        fig, ax = plt.subplots(figsize=(4, 4))
        pd.Series(sizes).plot(kind='barh', ax=ax, color='black') if plot else None
        ax.set_xscale('log')
        ax.invert_yaxis()
        plt.show()

    return _inData


def make_shareability_graph(_inData, params):
    """
    Prepares the shareability graphs from trip pairs
    :param _inData: inData.exmas.rides
    :return: inDara.exmas.S
    """
    rides = _inData.exmas.rides
    rides['degree'] = rides.apply(lambda x: len(x.indexes), axis=1)

    R2 = rides[rides.degree == 2].copy()
    R2['i'] = R2.indexes.apply(lambda x: x[0])  # for edge list
    R2['j'] = R2.indexes.apply(lambda x: x[1])
    R2 = R2.reset_index(drop=True)
    R2['index_copy'] = R2.index

    _inData.exmas.R[2] = R2
    # New part for weighting a graph:
    df = R2.copy()
    # df['weight'] = df['u_paxes'].apply(lambda x: norm.cdf(x[0], exmas_params.starting_probs.mu_prob, exmas_params.st_dev_prob)
    #                                            *norm.cdf(x[1], exmas_params.starting_probs.mu_prob, exmas_params.st_dev_prob))
    # df['weight'] = df['true_u_pax']
    df['weight'] = df['u_paxes']

    _inData.exmas.S = nx.from_pandas_edgelist(df, 'i', 'j',
                                              edge_attr=['kind', 'index_copy', 'weight'],
                                              create_using=nx.MultiDiGraph())  # create a graph
    return _inData


def enumerate_ride_extensions(r, S):
    """
    r rides
    S graph
    """
    ret = list()
    # find trips shareable with all trips of ride r
    S_r = None
    for t in r.indexes:  # iterate trips of ride r
        outs = set(S.neighbors(t))  # shareable trips with e
        S_r = outs if S_r is None else S_r & outs  # iterative intersection of trips
        if len(S_r) == 0:
            break  # early exit
    for q in S_r:  # iterate candidates
        E = [[S[e][q][i]['index_copy'] for i in list(S[e][q])] for e in
             r.indexes]  # list of (possibly) two edges connecting trips of r with q
        exts = list(product(*E))[0]
        if len(exts) > 0:
            ret.append(exts)
    return ret


# ALGORITHM 2/3
def extend_degree(_inData, params, degree):
    R = _inData.exmas.R

    # faster queries through dict
    dist_dict = _inData.exmas.requests.dist.to_dict()  # distances
    ttrav_dict = _inData.exmas.requests.ttrav.to_dict()  # travel times
    treq_dict = _inData.exmas.requests.treq.to_dict()  # requests times
    VoT_dict = _inData.exmas.requests.VoT.to_dict()  # values of time
    WtS_dict = _inData.exmas.requests.WtS.to_dict()
    panel_noise_dict = {i: _inData.prob.panel_noise[i] for i in
                        range(len(_inData.prob.panel_noise))}  # noise for utility

    nPotential = 0
    retR = list()  # for output

    for _, r in R[degree].iterrows():  # iterate through all rides to extend
        newtrips, nSearched = extend(r, _inData.exmas.S, R, params, degree, dist_dict, ttrav_dict, treq_dict, VoT_dict
                                     , WtS_dict, panel_noise_dict)
        retR.extend(newtrips)
        nPotential += nSearched

    df = pd.DataFrame(retR, columns=['indexes', 'indexes_orig', 'u_pax', 'u_veh', 'kind',
                                     'u_paxes', 'times', 'indexes_dest', 'true_u_pax',
                                     'true_u_paxes', 'delays'])  # data synthax for rides

    df = df[RIDE_COLS]
    df = df.reset_index()
    _inData.logger.info('At degree {} feasible extensions found out of {} searched'.format(degree,
                                                                                           df.shape[0],
                                                                                           nPotential))

    _inData.exmas.R[degree + 1] = df  # store output
    _inData.exmas.rides = pd.concat([_inData.exmas.rides, df], sort=False)
    if df.shape[0] > 0:
        assert_extension(_inData, params, degree + 1)

    return _inData


# ALGORITHM 2 a
def extend(r, S, R, params, degree, dist_dict, ttrav_dict, treq_dict, VoT_dict, WtS_dict, panel_noise_dict):
    """
    extends a single ride of a given degree with all feasible rides of degree+1
    calls trip_sharing_utility to test if ride is attractive
    :param r: shared ride
    :param S: graph
    :param R: all rides of this degree
    :param params:
    :param degree:
    :param dist_dict:
    :param ttrav_dict:
    :param treq_dict:
    :param VoT_dict:
    :return:
    """
    deptimefun = lambda dep: max([abs(dep + delay) ** 2 for delay in delays])  # minmax
    deptimefun = np.vectorize(deptimefun)
    accuracy = 10
    retR = list()
    potential = 0
    for extension in enumerate_ride_extensions(r, S):  # all possible extensions
        Eplus, Eminus, t, kind = list(), list(), list(), None
        E = dict()  # star extending r with q
        indexes_dest = r.indexes_dest.copy()
        potential += 1

        for trip in extension:  # E = Eplus + Eminus
            t = R[2].loc[trip]
            E[t.i] = t  # trips lookup table to determine times
            if t.kind == 20:  # to determine destination sequence
                Eplus.append(indexes_dest.index(t.i))  # FIFO
            elif t.kind == 21:
                Eminus.append(indexes_dest.index(t.i))  # LIFO

        q = t.j

        if len(Eminus) == 0:
            kind = 0  # pure FIFO
            pos = degree
        elif len(Eplus) == 0:
            kind = 1  # pure LIFO
            pos = 0
        else:
            if min(Eminus) > max(Eplus):
                pos = min(Eminus)
                kind = 2
            else:
                kind = -1

        if kind >= 0:  # feasible ride
            re = DotMap()  # new extended ride
            re.indexes = r.indexes + [q]
            re.indexes_orig = re.indexes
            indexes_dest.insert(pos, q)  # new destination order
            re.indexes_dest = indexes_dest

            # times[1] = oo, times[2] = od, times[3]=dd

            new_time_oo = [E[re.indexes_orig[-2]].times[1]]  # this is always the case

            if pos == degree:  # insert as last destination
                new_time_od = [E[re.indexes_dest[0]].times[2]]
                new_time_dd = [E[re.indexes_dest[-2]].times[3]]
                new_times = [r.times[0:degree] +
                             new_time_oo +
                             new_time_od +
                             r.times[degree + 1:] +
                             new_time_dd]

            elif pos == 0:  # insert as first destination
                new_times = [r.times[0:degree] +
                             E[re.indexes_orig[-2]].times[1:3] +
                             [E[re.indexes_dest[1]].times[3]] +
                             r.times[degree + 1:]]

            else:
                new_time_od = [E[re.indexes_dest[0]].times[2]]
                new_times = r.times[0:degree] + new_time_oo + new_time_od
                if len(r.times[degree + 1:degree + 1 + pos - 1]) > 0:  # not changed sequence before insertion
                    new_times += r.times[degree + 1:degree + 1 + pos - 1]  # only for degree>3

                # insertion
                new_times += [E[re.indexes_dest[pos - 1]].times[-1]]  # dd
                new_times += [E[re.indexes_dest[pos + 1]].times[-1]]  # dd

                if len(r.times[(degree + 1 + pos):]) > 0:  # not changed sequence after insertion
                    new_times += r.times[degree + 1 + pos:]  # only for degree>3

                new_times = [new_times]

            new_times = new_times[0]
            re.times = new_times

            # determine utilities
            dists = [dist_dict[_] for _ in re.indexes]  # distances
            ttrav_ns = [ttrav_dict[_] for _ in re.indexes]  # non shared travel times
            VoT = [VoT_dict[_] for _ in re.indexes]  # VoT of sharing travellers
            WtS = [WtS_dict[_] for _ in re.indexes]
            # shared travel times
            ttrav = [sum(new_times[i + 1:degree + 2 + re.indexes_dest.index(re.indexes[i])]) for i in
                     range(degree + 1)]

            # panel noise addon
            panel_noise = [panel_noise_dict[_] for _ in re.indexes]

            # first assume null delays
            feasible_flag = True
            noise = []
            for i in range(degree + 1):
                if params.get("noise", None) is None:
                    noise.append(0)
                else:
                    noise.append(np.random.normal(size=1, loc=params.noise.get("mean")))
                if trip_sharing_utility(params, dists[i], 0, ttrav[i], ttrav_ns[i], VoT[i], WtS[i]) + \
                        panel_noise[i] < 0:
                    feasible_flag = False
                    break
            if feasible_flag:
                # determine optimal departure time (if feasible with null delay)
                treq = [treq_dict[_] for _ in re.indexes]  # distances

                delays = [new_times[0] + sum(new_times[1:i]) - treq[i] for i in range(degree + 1)]

                dep_range = int(max([abs(_) for _ in delays]))

                if dep_range == 0:
                    x = [0]
                else:
                    x = np.arange(-dep_range, dep_range, min(dep_range, accuracy))
                d = (deptimefun(x))

                delays = [abs(_ + x[np.argmin(d)]) for _ in delays]
                # if _print:
                #    pd.Series(d, index=x).plot()  # option plot d=f(dep)
                u_paxes = list()
                true_u_paxes = list()
                noise = []

                for i in range(degree + 1):
                    true_u_paxes.append(
                        trip_sharing_utility(params, dists[i], delays[i], ttrav[i], ttrav_ns[i], VoT[i], WtS[i]))
                    if params.get("noise", None) is None:
                        u_paxes.append(true_u_paxes[-1] + panel_noise[i])
                    else:
                        noise.append(np.random.normal(loc=params.noise.get("mean", 0), scale=params.noise.get("st_dev", 0)))
                        u_paxes.append(true_u_paxes[-1] + panel_noise[i] + noise[-1])

                    if u_paxes[-1] < 0:
                        feasible_flag = False
                        break
                if feasible_flag:
                    re.true_u_paxes = [shared_trip_utility(params, dists[i], delays[i], ttrav[i], VoT[i], WtS[i]) for i
                                       in
                                       range(degree + 1)]
                    if params.get("noise", None) is None:
                        re.u_paxes = [x[0] - x[1] for x in zip(re.true_u_paxes, panel_noise)]
                    else:
                        re.u_paxes = [x[0] - x[1] - x[2] for x in zip(re.true_u_paxes, panel_noise, noise)]

                    re.pos = pos
                    re.times = new_times
                    re.u_pax = sum(re.u_paxes)
                    re.true_u_pax = sum(re.true_u_paxes)
                    re.u_veh = sum(re.times[1:])
                    re.delays = delays
                    if degree > 4:
                        re.kind = 100
                    else:
                        re.kind = 10 * (degree + 1) + kind
                    retR.append(dict(re))

    return retR, potential


def matching(_inData, params, plot=False, make_assertion=True):
    """
    called from the main loop
    :param _inData:
    :param plot:
    :param make_assertion: check if results are consistent
    :return: inData.exmas.schedule - selected rides (and keys to them in inData.exmas.requests)
    """
    rides = _inData.exmas.rides.copy()
    requests = _inData.exmas.requests.copy()

    rides['ttrav'] = rides.times.apply(lambda x: sum(x[1:]))
    rides['ttrav_ns'] = rides.indexes.apply(lambda x: sum([requests.iloc[t]['ttrav'] for t in x]))

    shared_rides = rides.loc[[len(t) > 1 for t in rides.indexes]].copy()
    shared_rides['saved_time'] = shared_rides["ttrav_ns"] - shared_rides["ttrav"]
    shared_rides['earnings_on_shared'] = shared_rides['ttrav_ns'].values * params.avg_speed * (1 - params.shared_discount)
    shared_rides['u_op'] = shared_rides['earnings_on_shared']/(shared_rides['saved_time']*params.avg_speed)

    opt_outs = False
    multi_platform_matching = params.get('multi_platform_matching', False)

    if not multi_platform_matching:  # classic matching for single platform
        selected = match(im=rides, r=requests, params=params, plot=plot,
                         make_assertion=make_assertion, logger=_inData.logger)
        rides['selected'] = pd.Series(selected)

    else:  # matching to multiple platforms
        # select only rides for which all travellers are assigned to this platform
        rides['platform'] = rides.apply(lambda row: list(set(_inData.exmas.requests.loc[row.indexes].platform.values)),
                                        axis=1)

        rides['platform'] = rides.platform.apply(lambda x: -2 if len(x) > 1 else x[0])
        rides['selected'] = 0

        opt_outs = -1 in rides.platform.unique()  # do we have travellers opting out

        for platform in rides.platform.unique():
            if platform >= 0:
                platform_rides = rides[rides.platform == platform]
                selected = match(im=platform_rides, r=requests[requests.platform == platform], params=params,
                                 plot=plot, make_assertion=False, logger=_inData.logger)

                rides['selected'].update(pd.Series(selected))

    schedule = rides[rides.selected == 1].copy()

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


def match(im, r, params={"matching_obj": "u_veh"}, plot=False, min_max="min", make_assertion=True, logger=None):
    """
    main call of bipartite matching on a graph
    :param im: possible rides
    :param r: requests
    :param params: parameter (including objective function)
    :param plot:
    :param make_assertion: test the results at the end
    :return: rides, selected rides, requests
    """
    request_indexes = dict()
    request_indexes_inv = dict()
    for i, index in enumerate(r.index.values):
        request_indexes[index] = i
        request_indexes_inv[i] = index

    im_indexes = dict()
    im_indexes_inv = dict()
    for i, index in enumerate(im.index.values):
        im_indexes[index] = i
        im_indexes_inv[i] = index

    # im['lambda_r'] = im.apply(
    #     lambda x: exmas_params.shared_discount if x.kind == 1 else 1 - x.u_veh / sum([r.loc[_].ttrav for _ in x.indexes]),
    #     axis=1)

    im['PassHourTrav_ns'] = im.apply(lambda x: sum([r.loc[_].ttrav for _ in x.indexes]), axis=1)

    try:
        r = r.reset_index()
    except ValueError:
        r = r.drop('index', axis=1)
        r = r.reset_index()

    # if exmas_params.profitability:
    #     im = im[im.lambda_r >= exmas_params.shared_discount]
    #     logger.info('Out of {} trips  {} are directly profitable.'.format(r.shape[0],
    #                                                                       im.shape[0])) if logger is not None else None

    nR = r.shape[0]

    def add_binary_row(r):
        ret = np.zeros(nR)
        for i in r.indexes:
            ret[request_indexes[i]] = 1
        return ret

    logger.info('Matching {} trips to {} rides in order to minimize {}'.format(nR,
                                                                               im.shape[0],
                                                                               params["matching_obj"])) if logger is not None else None
    im['row'] = im.apply(add_binary_row, axis=1)  # row to be used as constrain in optimization
    m = np.vstack(im['row'].values).T  # creates a numpy array for the constrains

    if plot:
        plt.rcParams['figure.figsize'] = [20, 3]
        plt.imshow(m, cmap='Greys', interpolation='nearest')
        # plt.spy(m, c = 'blue')
        plt.show()

    im['index'] = im.index.copy()

    im = im.reset_index(drop=True)

    # optimization
    if min_max == "min":
        prob = pulp.LpProblem("Matching problem", pulp.LpMinimize)  # problem
    else:
        prob = pulp.LpProblem("Matching problem", pulp.LpMaximize)

    variables = pulp.LpVariable.dicts("r", (i for i in im.index), cat='Binary')  # decision variables

    cost_col = params["matching_obj"]
    if cost_col == 'degree':
        costs = im.indexes.apply(lambda x: -(10 ** len(x)))
    else:
        costs = im[cost_col]  # set the costs

    prob += pulp.lpSum([variables[i] * costs[i] for i in variables]), 'ObjectiveFun'  # ffef

    j = 0  # adding constrains
    for imr in m:
        j += 1
        prob += pulp.lpSum([imr[i] * variables[i] for i in variables if imr[i] > 0]) == 1, 'c' + str(j)

    # solver = pulp.get_solver(solver_for_pulp())
    try:
        solver = pulp.get_solver(solver_for_pulp())
    except AttributeError:
        solver = pulp.getSolver(solver_for_pulp())

    solver.msg = False
    prob.solve(solver)  # main optimization call
    # prob.solve()  # main optimization call

    logger.info('Problem solution: {}. \n'
                'Total costs for single trips:  {:13,} '
                '\nreduced by matching to: {:20,}'.format(pulp.LpStatus[prob.status], int(sum(costs[:nR])),
                                                          int(pulp.value(
                                                              prob.objective)))) if logger is not None else None

    # assert pulp.value(prob.objective) <= sum(costs[:nR]) + 2  # we did not go above original

    locs = dict()
    for variable in prob.variables():
        i = int(variable.name.split("_")[1])

        locs[im_indexes_inv[i]] = (int(variable.varValue))
        # _inData.logger.info("{} = {}".format(int(variable.name.split("_")[1]), int(variable.varValue)))

    return locs


def evaluate_shareability(_inData, params, plot=False):
    """
    Calc KPIs for the results of assigning trips to shared rides
    :param _inData:
    :param params:
    :param plot:
    :return:
    """

    # plot
    ret = DotMap()
    r = _inData.exmas.requests.copy()

    schedule = _inData.exmas.schedule.copy()
    schedule['ttrav'] = schedule.apply(lambda x: sum(x.times[1:]), axis=1)

    fare = 0

    ret['VehHourTrav'] = schedule.ttrav.sum()
    ret['VehHourTrav_ns'] = r.ttrav.sum()

    ret['PassHourTrav'] = r.ttrav_sh.sum()
    ret['PassHourTrav_ns'] = r.ttrav.sum()

    ret['PassUtility'] = r.u_sh.sum()
    ret['PassUtility_ns'] = r.u.sum()

    # results['mean_ride_lambda'] = schedule.lambda_r.mean()
    ret['mean_lambda'] = 1 - schedule[schedule.kind > 1].u_veh.sum() / schedule[
        schedule.kind > 1].PassHourTrav_ns.sum()

    # results['shared_fares'] = schedule[schedule.kind > 1].PassHourTrav_ns.sum() * sp.price * (
    #       1 - sp.shared_discount)
    # results['full_fares'] = schedule[schedule.kind == 1].PassHourTrav_ns.sum() * sp.price
    ret['revenue_s'] = schedule.PassHourTrav_ns.sum() * params.price * (1 - params.shared_discount)
    ret['revenue_ns'] = schedule.PassHourTrav_ns.sum() * params.price
    ret['Fare_Discount'] = (ret['revenue_s'] - ret['revenue_ns']) / ret['revenue_ns']

    split = schedule.groupby('kind').sum()
    split['kind'] = split.index
    split['name'] = split.kind.apply(lambda x: SbltType(x).name)
    split = split.set_index('name')
    del split['kind']

    ret['nR'] = r.shape[0]

    ret['SINGLE'] = schedule[(schedule.kind == 1)].shape[0]
    ret['PAIRS'] = schedule[(schedule.kind > 1) & (schedule.kind < 30)].shape[0]
    ret['TRIPLES'] = schedule[(schedule.kind >= 30) & (schedule.kind < 40)].shape[0]
    ret['QUADRIPLES'] = schedule[(schedule.kind >= 40) & (schedule.kind < 50)].shape[0]
    ret['QUINTETS'] = schedule[(schedule.kind >= 50) & (schedule.kind < 100)].shape[0]
    ret['PLUS5'] = schedule[(schedule.kind == 100)].shape[0]
    ret['shared_ratio'] = 1 - ret['SINGLE'] / ret['nR']

    # df = pd.DataFrame(_inData.exmas.log.sizes).T[['potential', 'feasible']].reindex([1, 2, 3, 4])
    nR = r.shape[0]

    # df['selected'] = [results['SINGLE'], results['PAIRS'], results['TRIPLES'], results['QUADRIPLES']]
    # df['theoretical'] = [nR, nR ** 2, nR ** 3, nR ** 4]
    # _inData.exmas.log.sizes = df.fillna(0).astype(int)

    r['start'] = pd.to_datetime(r.treq, unit='s')
    r['end'] = pd.to_datetime(r.treq + r.ttrav, unit='s')
    fsns = fleet_size(r)
    ret['fleet_size_nonshared'] = max(fsns)

    schedule['start'] = pd.to_datetime([t[0] for t in schedule.times.values], unit='s')
    schedule['end'] = schedule.start + pd.to_timedelta([sum(t[1:]) for t in schedule.times.values], unit='s')
    fs = fleet_size(schedule)
    ret['fleet_size_shared'] = max(fs)
    if schedule[schedule.kind > 1].shape[0] > 0:
        shared_vehkm = schedule[schedule.kind > 1].u_veh.sum()
        ns_vehkm = r[r.kind > 1].ttrav.sum()
        ret['lambda_shared'] = 1 - shared_vehkm / ns_vehkm
    else:
        ret['lambda_shared'] = 0

    # results['fleet_size_shared'] = max(fs)
    if plot:
        fig, ax = plt.subplots()
        ax.set_ylabel("number of rides")
        ax.set_xlabel("time")

        fsns.plot(drawstyle='steps', ax=ax, label='non_shared', color='black')
        fs.plot(drawstyle='steps', ax=ax, label='shared', color='grey')
        ax.set_xticks([])
        plt.suptitle('fleet size proxy')
        plt.legend()

    _inData.logger.info(ret)
    _inData.exmas.res = pd.Series(ret)
    return _inData


#########
# UTILS #
#########

def trip_sharing_utility(params, dist, dep_delay, ttrav, ttrav_ns, VoT, WtS):
    # trip sharing utility for a trip, trips are shared only if this is positive.
    # difference
    return (params.price * dist / 1000 * params.shared_discount
            + VoT * (ttrav_ns - WtS * (ttrav + params.delay_value * abs(dep_delay))))


def shared_trip_utility(params, dist, dep_delay, ttrav, VoT, WtS):
    #  utility of a shared trip
    return (params.price * (1 - params.shared_discount) * dist / 1000 +
            VoT * WtS * (ttrav + params.delay_value * abs(dep_delay)))


def make_schedule(t, r):
    columns = ['node', 'times', 'req_id', 'od']
    degree = 2 * len(ast.literal_eval(t.indexes))
    df = pd.DataFrame(None, index=range(degree), columns=columns)
    x = ast.literal_eval(t.indexes_orig)
    s = [r.loc[i].origin for i in x] + [r.loc[i].destination for i in x]
    df.node = pd.Series(s)
    df.req_id = x + ast.literal_eval(t.indexes_dest)
    df.times = t.times
    df.od = pd.Series(['o'] * len(ast.literal_eval(t.indexes)) + ['d'] * len(ast.literal_eval(t.indexes)))
    return df


def init_log(sp, logger=None):
    level = sp.get('logger_level', "INFO")
    if level == 'INFO':
        level == logging.INFO
    elif level == 'WARNING':
        level == logging.WARNING
    elif level == 'CRITICAL':
        level = logging.CRITICAL
    if logger is None:
        logging.basicConfig(stream=sys.stdout, format='%(asctime)s-%(levelname)s-%(message)s',
                            datefmt='%d-%m-%y %H:%M:%S', level=level)

        logger = logging.getLogger()

        logger.setLevel(level)
        return logging.getLogger(__name__)
    else:
        logger.setLevel(level)
        return logger


def fleet_size(requests):
    requests = requests.sort_values('start')
    pickups = requests.set_index('start')
    pickups['starts'] = 1
    try:
        ret = pickups.resample('60s').sum().cumsum()[['starts']]
    except TypeError:
        temp_stamps = list(pickups.resample('60s'))
        ret = pd.DataFrame({'starts': np.cumsum([len(t[1]) for t in temp_stamps])}, index=[t[0] for t in temp_stamps])
    dropoffs = requests.set_index('end')
    dropoffs['ends'] = 1
    try:
        d = dropoffs.resample('60s').sum().ends.cumsum()
    except TypeError:
        temp_stamps = list(dropoffs.resample('60s'))
        d = pd.DataFrame({'ends': np.cumsum([len(t[1]) for t in temp_stamps])}, index=[t[0] for t in temp_stamps])
    ret = ret.join(d, how='outer')
    ret.starts = ret.starts.fillna(ret.starts.max())
    ret.ends = ret.ends.fillna(0)
    return ret.starts - ret.ends


def assert_extension(_inData, params, degree=3, nchecks=4, t=None):
    """
    Function checks whether all the resulting extended trips are coreectly calculated.
    Checks if ride travel times are in line with skim times.
    Used to debug, can be made silent or inactive for speed up (though it is definitely not a killer in performance)
    :param _inData:
    :param params:
    :param degree:
    :param nchecks:
    :param t:
    :return:
    """
    if t is None:
        rides = _inData.exmas.R[degree]
    else:
        rides = None
    the_skim = _inData.the_skim
    r = _inData.exmas.requests
    for _ in range(nchecks + 1):
        if t is None:
            t = rides.sample(1).iloc[0]
        os = t.indexes_orig
        ds = t.indexes_dest
        skim_times = list()
        degree = len(os)
        for i in range(degree - 1):
            o1 = r.loc[os[i]].origin
            o2 = r.loc[os[i + 1]].origin
            skim_times.append(the_skim.loc[o1, o2] + params.pax_delay)
        skim_times.append(the_skim.loc[r.loc[os[-1]].origin, r.loc[ds[0]].destination])

        for i in range(degree - 1):
            d1 = r.loc[ds[i]].destination
            d2 = r.loc[ds[i + 1]].destination
            skim_times.append(the_skim.loc[d1, d2])

        try:
            assert skim_times == t.times[1:]
            # if nchecks == 0:
            # _inData.logger.warning(skim_times, t.times[1:])
        except AssertionError as error:
            _inData.logger.critical('Assertion Error for extension')
            # _inData.logger.warning(t)
            _inData.logger.warning(params.pax_delay)
            _inData.logger.warning(skim_times)
            _inData.logger.warning(t.times[1:])
            assert skim_times == t.times[1:]


def add_noise(inData, params):
    if params.get("panel_noise", None) is not None:
        assert isinstance(params.panel_noise, dict), "Wrong type of panel_noise passed in json"
        seed = params.get('seed', None)
        if seed is not None:
            try:
                np.random.seed(int(params.seed))
            except:
                raise Exception('Passed seed cannot be set')
        inData.prob.panel_noise = np.random.normal(params.panel_noise.get('mean', 0),
                                                   params.panel_noise.get('st_dev', 0),
                                                   len(inData.requests))
    else:
        inData.prob.panel_noise = np.zeros(len(inData.requests))

    return inData


def sample_random_parameters(inData: DotMap, params: DotMap, sampling_func: FunctionType = lambda *args: None):
    """
    Function designed to
    @param sampling_func:
    @param inData:
    @param params:
    @return:
    """
    if inData.prob.get("sampled_random_parameters") is not None:
        return inData

    # if params.get("sampling_function_with_index", False) is False:
    #     params["sampling_function_with_index"] = False

    if params.get("distribution_variables", None) is None:
        inData.prob.sampled_random_parameters = pd.DataFrame()
        return inData

    if params.get("sampling_function", None) is not None:
        sampling_func = params.sampling_function

    seed = params.get('seed', None)
    if seed is not None:
        try:
            np.random.seed(int(params.seed))
        except:
            raise Exception('Passed seed cannot be set')

    type_of_distribution = params.get("type_of_distribution", None)
    number_of_requests = len(inData.requests)
    zeros = [0] * len(params.get("distribution_variables", []))

    if type_of_distribution == "discrete":
        assert isinstance(params.distribution_details, dict), "Incorrect format of distribution details - " \
                                                              "should be dict"
        randomised_variables = dict()

        for key in params.distribution_details.keys():
            randomised_variables[key] = np.random.choice(params.distribution_details[key][0], number_of_requests,
                                                         params.distribution_details[key][1])
        inData.prob.sampled_random_parameters = pd.DataFrame(randomised_variables)

    elif type_of_distribution == "manual" and sampling_func(*zeros) is not None:
        sample_from_interval = np.random.random([number_of_requests, len(zeros)])
        if not params.get("sampling_function_with_index", False):
            columns = params.distribution_variables
        else:
            columns = params.distribution_variables + ["class"]
        inData.prob.sampled_random_parameters = pd.DataFrame([sampling_func(*sample_from_interval[j, :])
                                                              for j in range(len(sample_from_interval))],
                                                             columns=columns)

    elif type_of_distribution == "multinormal":
        gen_func = mixed_discrete_norm_distribution(
            probs=params["multinormal_probs"],
            arguments=params["multinormal_args"],
            with_index=params.get("sampling_function_with_index", False)
        )
        sample_from_interval = np.random.random([number_of_requests, len(zeros)])
        if not params.get("sampling_function_with_index", False):
            columns = params.distribution_variables
        else:
            columns = params.distribution_variables + ["class"]
        inData.prob.sampled_random_parameters = pd.DataFrame([gen_func(*sample_from_interval[j, :])
                                                              for j in range(len(sample_from_interval))],
                                                             columns=columns)

    elif type_of_distribution == "normal":
        assert isinstance(params.distribution_details, dict), "Incorrect format of distribution details - " \
                                                              "should be dict"
        randomised_variables = dict()

        for key in params.distribution_details.keys():
            randomised_variables[key] = np.random.normal(size=number_of_requests,
                                                         loc=params.distribution_details[key][0],
                                                         scale=params.distribution_details[key][1])
        inData.prob.sampled_random_parameters = pd.DataFrame(randomised_variables)
    else:
        inData.prob.sampled_random_parameters = pd.DataFrame()

    if len(inData.prob.sampled_random_parameters) > 0:
        inData.prob.sampled_random_parameters.index = inData.requests.index

    return inData


def solver_for_pulp():
    system = platform.system()
    if system == "Windows":
        return "GLPK_CMD"
    else:
        return "PULP_CBC_CMD"


def extend_r_sampled_parameters(r, _inData, params):
    if "VoT" in _inData.prob.sampled_random_parameters.columns:
        new_temp_df = pd.merge(r['i'].copy(), _inData.prob.sampled_random_parameters['VoT'], left_on="i",
                               right_index=True)
        new_temp_df.index = r.index
        r.VoT_i = new_temp_df['VoT']

        new_temp_df = pd.merge(r['j'].copy(), _inData.prob.sampled_random_parameters['VoT'], left_on="j",
                               right_index=True)
        new_temp_df.index = r.index
        r.VoT_j = new_temp_df['VoT']
    else:
        r.VoT_i = params.VoT
        r.VoT_j = params.VoT

    if "WtS" in _inData.prob.sampled_random_parameters.columns:
        new_temp_df = pd.merge(r['i'].copy(), _inData.prob.sampled_random_parameters['WtS'], left_on="i",
                               right_index=True)
        new_temp_df.index = r.index
        r["WtS_i"] = new_temp_df['WtS']

        new_temp_df = pd.merge(r['j'].copy(), _inData.prob.sampled_random_parameters['WtS'], left_on="j",
                               right_index=True)
        new_temp_df.index = r.index
        r["WtS_j"] = new_temp_df['WtS']
    else:
        r["WtS_i"] = params.WtS
        r["WtS_j"] = params.WtS

    if "delay_value" in _inData.prob.sampled_random_parameters.columns:
        new_temp_df = pd.merge(r['i'].copy(), _inData.prob.sampled_random_parameters['delay_value'], left_on="i",
                               right_index=True)
        new_temp_df.index = r.index
        r["delay_value_i"] = new_temp_df['delay_value']

        new_temp_df = pd.merge(r['j'].copy(), _inData.prob.sampled_random_parameters['delay_value'], left_on="j",
                               right_index=True)
        new_temp_df.index = r.index
        r["delay_value_j"] = new_temp_df['delay_value']
    else:
        r["delay_value_i"] = params.delay_value
        r["delay_value_j"] = params.delay_value

    if "shared_discount" in _inData.prob.sampled_random_parameters.columns:
        new_temp_df = pd.merge(r['i'].copy(), _inData.prob.sampled_random_parameters['shared_discount'], left_on="i",
                               right_index=True)
        new_temp_df.index = r.index
        r["shared_discount_i"] = new_temp_df['shared_discount']

        new_temp_df = pd.merge(r['j'].copy(), _inData.prob.sampled_random_parameters['shared_discount'], left_on="j",
                               right_index=True)
        new_temp_df.index = r.index
        r["shared_discount_j"] = new_temp_df['shared_discount']
    else:
        r["shared_discount_i"] = params.shared_discount
        r["shared_discount_j"] = params.shared_discount

    return r


def utility_for_r(r, ij, params, sampled_noise, one_or_two):
    one_two = str(1) if one_or_two == 1 else str(2)
    if params.get("panel_noise", None) is not None:
        if params.get("noise", None) is not None:
            data_temp = np.random.normal(size=len(r), loc=params.noise["mean"], scale=params.noise["st_dev"])
            sampled_noise[str(ij) + "_utility_" + one_two] = pd.DataFrame(data_temp)
            sampled_noise[str(ij) + "_utility_" + one_two][["i", "j"]] = r[["i", "j"]].values.copy()
            sampled_noise[str(ij) + "_utility_" + one_two].set_index(["i", "j"], drop=True, inplace=True)
            r['utility_' + ij] = r['true_utility_' + ij] + r['panel_noise_' + ij] + \
                                 sampled_noise[ij + "_utility_" + one_two].iloc[:, 0].values
            r = r[r['utility_' + ij] > 0]
        else:
            r['utility_' + ij] = r['true_utility_' + ij] + r['panel_noise_' + ij]
            r = r[r['utility_' + ij] > 0]
    else:
        if params.get("noise", None) is not None:
            data_temp = np.random.normal(size=len(r), loc=params.noise["mean"], scale=params.noise["st_dev"])
            sampled_noise[str(ij) + "_utility_" + one_two] = pd.DataFrame(data_temp)
            sampled_noise[str(ij) + "_utility_" + one_two][["i", "j"]] = r[["i", "j"]].values.copy()
            sampled_noise[str(ij) + "_utility_" + one_two].set_index(["i", "j"], drop=True, inplace=True)
            r['utility_' + ij] = r['true_utility_' + ij] + sampled_noise[ij + "_utility_" + one_two].iloc[:, 0].values
            r = r[r['utility_' + ij] > 0]
        else:
            r['utility_' + ij] = r['true_utility_' + ij]
            r = r[r['utility_' + ij] > 0]
    return r, sampled_noise


def calculate_r_utility(r, ij, params, sampled_noise, one_or_two):
    one_two = str(1) if one_or_two == 1 else str(2)
    if params.get("panel_noise", None) is not None:
        if params.get("noise", None) is not None:
            r['u_' + str(ij)] = r['true_u_' + ij] - r['panel_noise_' + ij] \
                                - sampled_noise[ij + "_utility_" + one_two].loc[r.index, 0].values
        else:
            r['u_' + str(ij)] = r['true_u_' + ij] - r['panel_noise_' + ij]
    else:
        if params.get("noise", None) is not None:
            r['u_' + str(ij)] = r['true_u_' + ij] - sampled_noise[ij + "_utility_" + one_two].loc[r.index, 0].values
        else:
            r['u_' + str(ij)] = r['true_u_' + ij]
    return r


def check_if_correct_attributes(params):
    if params.get("distribution_variables", None) is not None:
        assert isinstance(params.distribution_variables, list), "distribution_variables should be a list"
        for j in params.distribution_variables:
            assert isinstance(j, str), "Elements of list exmas_params.distribution_variables should be str"
    if params.get("type_of_distribution", None) is not None:
        params.type_of_distribution = params.type_of_distribution.lower()
        assert params.type_of_distribution in ["discrete", "manual", "normal", "multinormal"], \
            "Incorrect type_of_distribution. Admissible: 'discrete', 'manual', 'normal', 'multinormal'"
        if params.type_of_distribution == "discrete":
            assert "distribution_details" in params.keys(), \
                "distribution_details must be provided for discrete distribution. Example:" \
                '"distribution_details": {"VoT": [[0.0035, 0.003, 0.004], [0.6, 0.2, 0.2]]}'
            assert isinstance(params.distribution_details, dict), \
                'Incorrect type of distribution_details. Correct form: ' \
                'distribution_details": {"VoT": [[0.0035, 0.003, 0.004], [0.6, 0.2, 0.2]]}'
        if params.type_of_distribution == "normal":
            assert "distribution_details" in params.keys(), \
                "distribution_details must be provided for normal distribution (mean, st_dev)"
            assert {"mean", "st_dev"}.issubset(set(params.distribution_details.keys())), \
                "distribution_details: {mean: x, st_dev: y} is the ony admissible form."
    if params.get("noise", None) is not None:
        assert {"mean", "st_dev"}.issubset(set(params.noise.keys())), "Incorrect noise (should be dict: mean, st_dev)"
    if params.get("panel_noise", None) is not None:
        assert {"mean", "st_dev"}.issubset(set(params.panel_noise.keys()))
