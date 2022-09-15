import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from collections import namedtuple

import Topology.utils_topology as utils
import NYC_tools.NYC_data_prep_functions as nyc_tools
from ExMAS.main_prob import main as exmas_algo


if __name__ == "__main__":
    """ Load all the topological parameters """
    config = utils.get_parameters('configs/config_olha.json')

    """ Set up varying parameters (optional) """
    # config.variable = 'shared_discount'
    # config.values = [0.22, 0.24]

    """ Run parameters """
    config.replications = 1
    config.no_batches = 3

    """ Prepare folder """
    utils.create_results_directory(config)

    """ Prepare data """
    dotmaps_list, params = nyc_tools.prepare_batches(config.no_batches,
                                                     filter_function=lambda x: len(x.requests) > 0,
                                                     config=config.initial_parameters)

    """ Run ExMAS """
    dotmaps_list_results, settings_list = nyc_tools.testing_exmas_basic(exmas_algo, params, dotmaps_list,
                                                                          topo_params=config,
                                                                          replications=config.replications,
                                                                          logger_level='INFO')

    final_results = zip([x.sblts.requests for x in dotmaps_list_results],
                        [x.sblts.schedule for x in dotmaps_list_results],
                        [x.sblts.res for x in dotmaps_list_results],
                        settings_list)
    final_results = [{"requests": x[0],
                      "schedule": x[1],
                      "results": x[2],
                      "settings": x[3]} for x in final_results]
    utils.save_with_pickle(final_results, 'final_res', config)
