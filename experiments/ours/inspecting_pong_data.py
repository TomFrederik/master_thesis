import numpy as np
import matplotlib.pyplot as plt


import sys
sys.path.append('../../')

from src.common.datasets import construct_pong_train_val_data


train, val = construct_pong_train_val_data('../pong_data.hdf5', percentage=1, num_views=1, dropout=0, null_value=1, max_len=10, get_player_pos=False)

print(len(train))
obs = train[0][0]

for frame in obs:
    plt.figure()
    plt.imshow(frame[0], cmap='gray')
    plt.show()
    plt.close()