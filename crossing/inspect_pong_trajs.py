import numpy as np
import matplotlib.pyplot as plt

data_file = "ppo_pong_experience.npz"
data = np.load(data_file, allow_pickle=True)
obs = data['obs']
rewards = data['rewards']
dones = data['done']
actions = data['action']

print(rewards)
print(dones)

# for rew in rewards:
#     print(rew)


# for frame in obs:
#     plt.imshow(frame[0], cmap='gray')
#     plt.show()
#     plt.close()