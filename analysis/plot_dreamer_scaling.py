import matplotlib.pyplot as plt
import json
import numpy as np

JOB_ID = "9826900"

SETTINGS = [4, 8, 12, 16, 20, 24, 28]

for loss in ['tuning_loss', 'value_prefix_loss', 'kl_loss']:
    sparse_file = f"./data/dreamer/{JOB_ID}/{loss}/mean_std_per_setting.json"
    suffix = {'tuning_loss': 'tune', 'value_prefix_loss':'rew', 'kl_loss':'dyn'}[loss]
    title = {'tuning_loss': 'Tuning', 'value_prefix_loss':'Rew', 'kl_loss':'Dyn'}[loss]
    with open(sparse_file, "r") as f:
        mean_std_per_setting = json.load(f)

    xticks = SETTINGS
    plot_kwargs = dict(width=0.025, capsize=4)

    val = [mean_std_per_setting[str(i)][0] for i in SETTINGS]
    err = [mean_std_per_setting[str(i)][1] for i in SETTINGS]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    plt.suptitle("Sparse scaling")
    ax.errorbar(xticks, val, yerr=err, label="sparse", capsize=plot_kwargs['capsize'])
    ax.set_xticks(xticks, xticks)
    ax.grid(linestyle='--', alpha=0.5)
    ax.set_xlabel(r"Num Vars $N$")
    ax.set_ylabel(f"{title} loss")
    ax.legend()
    plt.savefig(f"dreamer_scaling_{suffix}.pdf")
