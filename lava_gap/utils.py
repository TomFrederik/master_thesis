from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def display_helper(
    obs:Dict, 
    save_path:Optional[str] = None
) -> None:
    num_keys = len(obs.keys())
    fig, axes = plt.subplots(1, num_keys, sharey=True, )
    
    for i, key in enumerate(obs.keys()):
        if obs[key] is None:
            image = np.zeros((9,9), dtype=np.uint8) #TODO this image size is currently hard coded
        else:
            image = obs[key].transpose(1,0)

        axes[i].imshow(image, cmap='inferno')
        axes[i].set_title(key)
        axes[i].set_axis_off()

    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0, wspace=0, hspace=0)
    # plt.margins(0,0)
    if save_path is not None:
        plt.savefig(save_path, pad_inches=0.1, bbox_inches='tight')
    plt.show()