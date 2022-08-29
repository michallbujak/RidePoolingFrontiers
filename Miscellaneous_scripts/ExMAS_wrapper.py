import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import Topology.utils_topology as utils
import NYC_tools.NYC_data_prep_functions as nyc_tools
from ExMAS.main_prob_coeffs import main as exmas_algo

import scipy.stats as ss


if __name__ == "__main__":
    """ Load all the topological parameters """
    config = utils.get_parameters('configs/test_config.json')

    """ Set up varying parameters (optional) """
    # config.variable = 'shared_discount'
    # config.values = [0.22, 0.24]

    """ Run parameters """
    config.replications = 1
    config.no_batches = 1

    """ Prepare folder """
    utils.create_results_directory(config)

    """ Prepare data """
    dotmaps_list, params = nyc_tools.prepare_batches(config.no_batches,
                                                     filter_function=lambda x: len(x.requests) > 0,
                                                     config=config.initial_parameters)

    """ Run ExMAS """
    params = utils.update_probabilistic(config, params)
    params.sampling_function = utils.inverse_normal([0.0035, 1.3], [0.0005, 0.1])

    dotmaps_list_results, settings_list = nyc_tools.testing_exmas_basic(exmas_algo, params, dotmaps_list,
                                                                          topo_params=config,
                                                                          replications=config.replications,
                                                                          logger_level='INFO')

    final_results = zip([x.sblts.res for x in dotmaps_list_results], settings_list)
    utils.save_with_pickle(final_results, 'final_res', config)
