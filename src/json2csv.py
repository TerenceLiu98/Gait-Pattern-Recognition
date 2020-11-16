import pandas as pd
import numpy as np

import os 
path = '../data/王/第二组/'
files = os.listdir(path)[:-1]

WANG = pd.DataFrame()
WANG['timestamp'] = pd.read_json(path + files[0])['timestamp']

for i in range(len(files)):
    tmp = pd.read_json(path + files[i])
    WANG[files[i][:-4]] = tmp['value']
