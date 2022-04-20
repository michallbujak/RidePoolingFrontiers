from ExMAS.main_prob import main as exmas_algo
from NYC_tools import NYC_data_prep_functions as nyc_func

idas, params = nyc_func.prepare_batches(2, filter_function=lambda x: len(x.requests) > 20, config_name="nyc_prob")

res = nyc_func.run_exmas_nyc_batches(exmas_algo, params, idas, replications=3)

print([x.sblts.res[['PassUtility', 'PassUtility_ns']] for x in res])
