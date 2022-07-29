import matplotlib.pyplot as plt
import json
import numpy as np

JOB_ID = "9770135"
SETTING = "8"


prior_file_dir = f"./data/dense/{JOB_ID}/prior_entropy/mean_std_per_setting.json"
post_file_dir = f"./data/dense/{JOB_ID}/posterior_entropy/mean_std_per_setting.json"

with open(prior_file_dir, "r") as f:
    prior_mean_std_per_setting = json.load(f)
    
with open(post_file_dir, "r") as f:
    post_mean_std_per_setting = json.load(f)


prior_data = prior_mean_std_per_setting[SETTING]
post_data = post_mean_std_per_setting[SETTING]


fig, ax = plt.subplots(figsize=(10,5))
ax.errorbar(range(len(prior_data)), prior_data, yerr=prior_data, label="Prior", color="blue", capsize=3)
ax.errorbar(range(len(post_data)), post_data, yerr=post_data, label="Posterior", color="red", capsize=3)
ax.set_xlabel("Epoch")
ax.set_ylabel("Entropy")
plt.show()