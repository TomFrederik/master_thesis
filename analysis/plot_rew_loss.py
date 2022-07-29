import matplotlib.pyplot as plt
import json
import numpy as np

OURS_JOB_ID = "9770135"
DREAMER_JOB_ID = "9801838"

no_05 = [2, 4, 6]
yes_05 = [0, 1, 3, 5]
no_075 = [9, 11, 13]
yes_075 = [7, 8, 10, 12]
no_1 = [16, 18, 20]
yes_1 = [14, 15, 17, 19]



ours_file = f"./data/ours/{OURS_JOB_ID}/value_prefix_loss/mean_std_per_setting.json"
dreamer_file = f"./data/dreamer/{DREAMER_JOB_ID}/value_prefix_loss/mean_std_per_setting.json"

with open(ours_file, "r") as f:
    ours_mean_std_per_setting = json.load(f)
    
with open(dreamer_file, "r") as f:
    dreamer_mean_std_per_setting = json.load(f)


data = ours_mean_std_per_setting

ours_test_only_true_view_075 = [ours_mean_std_per_setting[str(key)] for key in yes_075]
ours_test_only_true_view_05 = [ours_mean_std_per_setting[str(key)] for key in yes_05]
ours_test_only_true_view_1 = [ours_mean_std_per_setting[str(key)] for key in yes_1]

ours_test_only_false_view_05 = [ours_mean_std_per_setting[str(key)] for key in no_05]
ours_test_only_false_view_075 = [ours_mean_std_per_setting[str(key)] for key in no_075]
ours_test_only_false_view_1 = [ours_mean_std_per_setting[str(key)] for key in no_1]

dreamer_test_only_true_view_075 = [dreamer_mean_std_per_setting[str(key)] for key in yes_075]
dreamer_test_only_true_view_05 = [dreamer_mean_std_per_setting[str(key)] for key in yes_05]
dreamer_test_only_true_view_1 = [dreamer_mean_std_per_setting[str(key)] for key in yes_1]

dreamer_test_only_false_view_05 = [dreamer_mean_std_per_setting[str(key)] for key in no_05]
dreamer_test_only_false_view_075 = [dreamer_mean_std_per_setting[str(key)] for key in no_075]
dreamer_test_only_false_view_1 = [dreamer_mean_std_per_setting[str(key)] for key in no_1]


xticks_yes = np.array([0, 0.1, 0.2, 0.5])
xticks_no = np.array([0.1, 0.2, 0.5])
plot_kwargs = dict(width=0.025, capsize=4)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), sharey=True)
# ax.plot(xticks_yes, [x[0] for x in test_only_true_view_075], label=r"$p=0.75$")
ax[0].errorbar(xticks_yes, [x[0] for x in ours_test_only_true_view_05], yerr=[x[1] for x in ours_test_only_true_view_05], label="Ours", capsize=plot_kwargs['capsize'])
ax[1].errorbar(xticks_yes, [x[0] for x in ours_test_only_true_view_075], yerr=[x[1] for x in ours_test_only_true_view_075], label="Ours", capsize=plot_kwargs['capsize'])
ax[2].errorbar(xticks_yes, [x[0] for x in ours_test_only_true_view_1], yerr=[x[1] for x in ours_test_only_true_view_1], label="Ours", capsize=plot_kwargs['capsize'])
ax[0].errorbar(xticks_yes, [x[0] for x in dreamer_test_only_true_view_05], yerr=[x[1] for x in dreamer_test_only_true_view_05], label="Dreamer", capsize=plot_kwargs['capsize'])
ax[1].errorbar(xticks_yes, [x[0] for x in dreamer_test_only_true_view_075], yerr=[x[1] for x in dreamer_test_only_true_view_075], label="Dreamer", capsize=plot_kwargs['capsize'])
ax[2].errorbar(xticks_yes, [x[0] for x in dreamer_test_only_true_view_1], yerr=[x[1] for x in dreamer_test_only_true_view_1], label="Dreamer", capsize=plot_kwargs['capsize'])

plt.suptitle("Test_only_dropout = True")
plt.yscale("log")
plt.ylim(1e-9)
ax[0].set_title('p=0.5')
ax[1].set_title('p=0.75')
ax[2].set_title('p=1')
ax[0].set_xlabel(r"Dropout probability $d$")
ax[1].set_xlabel(r"Dropout probability $d$")
ax[2].set_xlabel(r"Dropout probability $d$")
ax[0].set_ylabel("Reward loss")
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), sharey=True)
# ax.plot(xticks_yes, [x[0] for x in test_only_true_view_075], label=r"$p=0.75$")
ax[0].errorbar(xticks_no, [x[0] for x in ours_test_only_false_view_05], yerr=[x[1] for x in ours_test_only_false_view_05], label="Ours", capsize=plot_kwargs['capsize'])
ax[1].errorbar(xticks_no, [x[0] for x in ours_test_only_false_view_075], yerr=[x[1] for x in ours_test_only_false_view_075], label="Ours", capsize=plot_kwargs['capsize'])
ax[2].errorbar(xticks_no, [x[0] for x in ours_test_only_false_view_1], yerr=[x[1] for x in ours_test_only_false_view_1], label="Ours", capsize=plot_kwargs['capsize'])
ax[0].errorbar(xticks_no, [x[0] for x in dreamer_test_only_false_view_05], yerr=[x[1] for x in dreamer_test_only_false_view_05], label="Dreamer", capsize=plot_kwargs['capsize'])
ax[1].errorbar(xticks_no, [x[0] for x in dreamer_test_only_false_view_075], yerr=[x[1] for x in dreamer_test_only_false_view_075], label="Dreamer", capsize=plot_kwargs['capsize'])
ax[2].errorbar(xticks_no, [x[0] for x in dreamer_test_only_false_view_1], yerr=[x[1] for x in dreamer_test_only_false_view_1], label="Dreamer", capsize=plot_kwargs['capsize'])

plt.suptitle("Test_only_dropout = False")
plt.yscale("log")
plt.ylim(1e-9)
ax[0].set_title('p=0.5')
ax[1].set_title('p=0.75')
ax[2].set_title('p=1')
ax[0].set_xlabel(r"Dropout probability $d$")
ax[1].set_xlabel(r"Dropout probability $d$")
ax[2].set_xlabel(r"Dropout probability $d$")
ax[0].set_ylabel("Reward loss")
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()