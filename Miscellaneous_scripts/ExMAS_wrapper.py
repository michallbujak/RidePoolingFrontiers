import os

os.chdir(os.path.dirname(os.getcwd()))

from ExMAS.main_prob import main
import NYC_tools.utils_nyc as utils_nyc

params = utils_nyc.get_config('ExMAS/data/configs/nyc_prob.json')
# inData = load_G(inData, params, stats=True)
# inData = utils_nyc.load_nyc_csv(inData, params)
# inData = utils_nyc.load_G(inData, params)

import pickle
# with open('test_obj.obj', 'wb') as file:
#     pickle.dump(inData, file)
with open('test_obj.obj', 'rb') as file:
  inData = pickle.load(file)
inData = main(inData, params, plot = False)

print(inData.sblts.res)
