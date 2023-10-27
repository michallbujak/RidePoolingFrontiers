from dotmap import DotMap
from tqdm import tqdm


def exmas_loop_func(
        exmas_algorithm,
        params,
        list_databanks,
        topo_params=DotMap({'variable': None}),
        replications=1,
        logger=None,
        sampling_function_with_index=False
):
    results = []
    settings = []
    params.logger_level = "CRITICAL"

    def log_func(log, msg):
        if log is None:
            print(msg)
        else:
            log.info(msg)

    log_func(logger, " Calculating ExMAS values \n ")

    params.sampling_function_with_index = sampling_function_with_index
    pbar = tqdm(total=len(list_databanks) * replications)

    for i in range(len(list_databanks)):
        log_func(logger, " Batch no. " + str(i))
        step = 0

        for j in range(replications):
            pbar.update(1)
            if topo_params.get("variable", None) is None:
                temp = exmas_algorithm(list_databanks[i], params, False)
                results.append(temp.copy())
                step += 1
                settings.append({'Replication_ID': j, 'Batch': i})
                log_func(logger, 'Attached batch number: ' + str(i))

            else:
                for k in range(len(topo_params['values'])):
                    params[topo_params['variable']] = topo_params['values'][k]
                    temp = exmas_algorithm(list_databanks[i], params, False)
                    results.append(temp.copy())
                    settings.append({
                        'Replication': j,
                        'Batch': i,
                        topo_params.variable: topo_params['values'][k],
                        'Start_time': list_databanks[i].requests.iloc[0, ]['pickup_datetime'],
                        'End_time': list_databanks[i].requests.iloc[-1, ]['pickup_datetime'],
                        'Demand_size': len(list_databanks[i].requests)
                    })
                    log_func(logger, 'Attached batch number: ' + str(i))

    pbar.close()
    log_func(logger, f"Number of calculated results for batches is: {len(results)}")
    log_func(logger, "ExMAS calculated \n")

    return results, settings
