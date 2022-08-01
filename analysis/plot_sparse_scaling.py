import matplotlib.pyplot as plt
import json
import numpy as np

SPARSE_JOB_ID = "9820227"

for loss in ['tuning_loss', 'value_prefix_loss', 'dyn_loss']:
    sparse_file = f"./data/sparse/{SPARSE_JOB_ID}/{loss}/mean_std_per_setting.json"
    suffix = {'tuning_loss': 'tune', 'value_prefix_loss':'rew', 'dyn_loss':'dyn'}[loss]
    title = {'tuning_loss': 'Tuning', 'value_prefix_loss':'Rew', 'dyn_loss':'Dyn'}[loss]
    with open(sparse_file, "r") as f:
        sparse_mean_std_per_setting = json.load(f)

    xticks = np.arange(10,17)
    plot_kwargs = dict(width=0.025, capsize=4)

    val = [sparse_mean_std_per_setting[str(i)][0] for i in range(10, 17)]
    err = [sparse_mean_std_per_setting[str(i)][1] for i in range(10, 17)]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    plt.suptitle("Sparse scaling")
    ax.errorbar(xticks, val, yerr=err, label="sparse", capsize=plot_kwargs['capsize'])
    ax.set_xticks(xticks, xticks)
    ax.grid(linestyle='--', alpha=0.5)
    ax.set_xlabel(r"Num Vars $N$")
    ax.set_ylabel(f"{title} loss")
    ax.legend()
    plt.savefig(f"sparse_scaling_{suffix}.pdf")
