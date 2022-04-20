import json
from dotmap import DotMap
import pandas as pd


def get_parameters(path, time_correction=False):
    with open(path) as json_file:
        data = json.load(json_file)
        config = DotMap(data)
    if time_correction:
        config['t0'] = pd.Timestamp('15:00')

    return config