from tqdm import tqdm

from Individual_pricing.pricing_utils.batch_preparation import log_func


def exmas_loop_func(
        exmas_algorithm,
        exmas_params,
        list_databanks,
        **kwargs
):
    results = []
    settings = []
    exmas_params.logger_level = "CRITICAL"

    log_func(20, "Calculating ExMAS values", kwargs.get('logger'))

    exmas_params.sampling_function_with_index = False
    pbar = tqdm(total=len(list_databanks))

    for i in range(len(list_databanks)):
        log_func(10, "Batch no. " + str(i), kwargs.get('logger'))
        if type(list_databanks) == list:
            _data = list_databanks[i]
        else:
            _data = list_databanks
        if type(_data) == list:
            _data = _data[0]
        temp = exmas_algorithm(_data,
                               exmas_params,
                               False)
        results.append(temp.copy())
        if kwargs.get("return_settings", True):
            settings.append({'Batch': i,
                             'Start_time': temp.requests.iloc[0, ]['pickup_datetime'],
                             'End_time': temp.requests.iloc[-1, ]['pickup_datetime'],
                             'Demand_size': len(temp.requests)})

        log_func(10, 'Attached batch number: ' + str(i), kwargs.get('logger'))
        pbar.update(1)
    pbar.close()

    log_func(20, f"Number of calculated results for batches is: {len(results)}",
             kwargs.get('logger'))

    if kwargs.get("return_settings", False):
        return results, settings
    else:
        return results
