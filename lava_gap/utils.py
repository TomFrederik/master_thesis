from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def display_helper(obs:Dict) -> None:
    num_keys = len(obs.keys())
    fig, axes = plt.subplots(1, num_keys, sharey=True)
    
    for i, key in enumerate(obs.keys()):
        if obs[key] is None:
            image = np.zeros((9,9), dtype=np.uint8) #TODO this image size is currently hard coded
        else:
            image = obs[key].transpose(1,0)

        axes[i].imshow(image)
        axes[i].set_title(key)
        axes[i].set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.show()