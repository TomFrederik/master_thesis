import json
import os
from itertools import product

params = {
    'percentage': [0.5,0.75,1.0],
    'dropout': [0,0.1,0.2,0.5],
    'test_only_dropout': ['yes','no']
}

os.makedirs('./hparam_files/hmm/perc_dropout', exist_ok=True)

ctr = 0
for i, results in enumerate(product(*params.values())):
    new_params = {k: v for k, v in zip(params.keys(), results)}
    if new_params['dropout'] == 0 and new_params['test_only_dropout'] == 'no':
        continue
    else:
        print(new_params)
        json.dump(new_params, open(f'./hparam_files/hmm/perc_dropout/{ctr}.json', 'w'))
        ctr += 1