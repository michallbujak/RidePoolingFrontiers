import os
import pandas as pd
import random
import osmnx as ox

os.chdir(os.path.dirname(os.getcwd()))

from ExMAS.main_prob import main
from ExMAS.utils import get_config
from ExMAS.utils import load_G
import ExMAS.utils_nyc as utils_nyc
from ExMAS.utils import inData as inData

params = utils_nyc.get_config('ExMAS/data/configs/nyc_prob.json')
inData = load_G(inData, params, stats=True)
inData = utils_nyc.load_nyc_csv(inData, params)
inData = utils_nyc.load_G(inData, params)

inData = main(inData, params, plot = True)

print(inData.sblts.res)
