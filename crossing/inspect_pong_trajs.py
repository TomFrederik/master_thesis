import numpy as np
import matplotlib.pyplot as plt

data_file = "ppo_pong_experience.npz"
data = np.load(data_file)
obs = data['obs']

for frame in obs:
    plt.imshow(frame[0], cmap='gray')
    plt.show()
    plt.close()