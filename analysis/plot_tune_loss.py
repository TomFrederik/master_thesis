import matplotlib.pyplot as plt
import json
import numpy as np

DENSE_JOB_ID = "9770135"
SPARSE_JOB_ID = "9807092"
DREAMER_JOB_ID = "9801838"

no_05 = [2, 4, 6]
yes_05 = [0, 1, 3, 5]
no_075 = [9, 11, 13]
yes_075 = [7, 8, 10, 12]
no_1 = [16, 18, 20]
yes_1 = [14, 15, 17, 19]



dense_file = f"./data/dense/{DENSE_JOB_ID}/tuning_loss/mean_std_per_setting.json"
sparse_file = f"./data/sparse/{SPARSE_JOB_ID}/tuning_loss/mean_std_per_setting.json"
dreamer_file = f"./data/dreamer/{DREAMER_JOB_ID}/tuning_loss/mean_std_per_setting.json"

with open(dense_file, "r") as f:
    dense_mean_std_per_setting = json.load(f)
    
with open(sparse_file, "r") as f:
    sparse_mean_std_per_setting = json.load(f)
    
with open(dreamer_file, "r") as f:
    dreamer_mean_std_per_setting = json.load(f)


dense_test_only_true_view_075 = [dense_mean_std_per_setting[str(key)] for key in yes_075]
dense_test_only_true_view_05 = [dense_mean_std_per_setting[str(key)] for key in yes_05]
dense_test_only_true_view_1 = [dense_mean_std_per_setting[str(key)] for key in yes_1]

dense_test_only_false_view_05 = [dense_mean_std_per_setting[str(key)] for key in no_05]
dense_test_only_false_view_075 = [dense_mean_std_per_setting[str(key)] for key in no_075]
dense_test_only_false_view_1 = [dense_mean_std_per_setting[str(key)] for key in no_1]

sparse_test_only_true_view_075 = [sparse_mean_std_per_setting[str(key)] for key in yes_075]
sparse_test_only_true_view_05 = [sparse_mean_std_per_setting[str(key)] for key in yes_05]
sparse_test_only_true_view_1 = [sparse_mean_std_per_setting[str(key)] for key in yes_1]

sparse_test_only_false_view_05 = [sparse_mean_std_per_setting[str(key)] for key in no_05]
sparse_test_only_false_view_075 = [sparse_mean_std_per_setting[str(key)] for key in no_075]
sparse_test_only_false_view_1 = [sparse_mean_std_per_setting[str(key)] for key in no_1]

dreamer_test_only_true_view_075 = [dreamer_mean_std_per_setting[str(key)] for key in yes_075]
dreamer_test_only_true_view_05 = [dreamer_mean_std_per_setting[str(key)] for key in yes_05]
dreamer_test_only_true_view_1 = [dreamer_mean_std_per_setting[str(key)] for key in yes_1]

dreamer_test_only_false_view_05 = [dreamer_mean_std_per_setting[str(key)] for key in no_05]
dreamer_test_only_false_view_075 = [dreamer_mean_std_per_setting[str(key)] for key in no_075]
dreamer_test_only_false_view_1 = [dreamer_mean_std_per_setting[str(key)] for key in no_1]


xticks_yes = np.array([0, 0.1, 0.2, 0.5])
xticks_no = np.array([0.1, 0.2, 0.5])
plot_kwargs = dict(width=0.025, capsize=4)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True, gridspec_kw=dict(wspace=0.05))
# ax.plot(xticks_yes, [x[0] for x in test_only_true_view_075], label=r"$p=0.75$")
ax[0].errorbar(xticks_yes, [x[0] for x in dense_test_only_true_view_05], yerr=[x[1] for x in dense_test_only_true_view_05], label="dense", capsize=plot_kwargs['capsize'])
ax[1].errorbar(xticks_yes, [x[0] for x in dense_test_only_true_view_075], yerr=[x[1] for x in dense_test_only_true_view_075], label="dense", capsize=plot_kwargs['capsize'])
ax[2].errorbar(xticks_yes, [x[0] for x in dense_test_only_true_view_1], yerr=[x[1] for x in dense_test_only_true_view_1], label="dense", capsize=plot_kwargs['capsize'])

ax[0].errorbar(xticks_yes, [x[0] for x in sparse_test_only_true_view_05], yerr=[x[1] for x in sparse_test_only_true_view_05], label="sparse", capsize=plot_kwargs['capsize'])
ax[1].errorbar(xticks_yes, [x[0] for x in sparse_test_only_true_view_075], yerr=[x[1] for x in sparse_test_only_true_view_075], label="sparse", capsize=plot_kwargs['capsize'])
ax[2].errorbar(xticks_yes, [x[0] for x in sparse_test_only_true_view_1], yerr=[x[1] for x in sparse_test_only_true_view_1], label="sparse", capsize=plot_kwargs['capsize'])

ax[0].errorbar(xticks_yes, [x[0] for x in dreamer_test_only_true_view_05], yerr=[x[1] for x in dreamer_test_only_true_view_05], label="Dreamer", capsize=plot_kwargs['capsize'])
ax[1].errorbar(xticks_yes, [x[0] for x in dreamer_test_only_true_view_075], yerr=[x[1] for x in dreamer_test_only_true_view_075], label="Dreamer", capsize=plot_kwargs['capsize'])
ax[2].errorbar(xticks_yes, [x[0] for x in dreamer_test_only_true_view_1], yerr=[x[1] for x in dreamer_test_only_true_view_1], label="Dreamer", capsize=plot_kwargs['capsize'])

plt.suptitle("Test_only_dropout = True")
ax[0].set_title('p=0.5')
ax[1].set_title('p=0.75')
ax[2].set_title('p=1')
ax[0].set_xlabel(r"Dropout probability $d$")
ax[1].set_xlabel(r"Dropout probability $d$")
ax[2].set_xlabel(r"Dropout probability $d$")
ax[0].set_xticks(xticks_yes, xticks_yes)
ax[1].set_xticks(xticks_yes, xticks_yes)
ax[2].set_xticks(xticks_yes, xticks_yes)

ax[0].set_ylabel("Tuning loss")
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.savefig("test_only_true.pdf")

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True, gridspec_kw=dict(wspace=0.05))
# ax.plot(xticks_yes, [x[0] for x in test_only_true_view_075], label=r"$p=0.75$")
ax[0].errorbar(xticks_no, [x[0] for x in dense_test_only_false_view_05], yerr=[x[1] for x in dense_test_only_false_view_05], label="dense", capsize=plot_kwargs['capsize'])
ax[1].errorbar(xticks_no, [x[0] for x in dense_test_only_false_view_075], yerr=[x[1] for x in dense_test_only_false_view_075], label="dense", capsize=plot_kwargs['capsize'])
ax[2].errorbar(xticks_no, [x[0] for x in dense_test_only_false_view_1], yerr=[x[1] for x in dense_test_only_false_view_1], label="dense", capsize=plot_kwargs['capsize'])

ax[0].errorbar(xticks_no, [x[0] for x in sparse_test_only_false_view_05], yerr=[x[1] for x in sparse_test_only_false_view_05], label="sparse", capsize=plot_kwargs['capsize'])
ax[1].errorbar(xticks_no, [x[0] for x in sparse_test_only_false_view_075], yerr=[x[1] for x in sparse_test_only_false_view_075], label="sparse", capsize=plot_kwargs['capsize'])
ax[2].errorbar(xticks_no, [x[0] for x in sparse_test_only_false_view_1], yerr=[x[1] for x in sparse_test_only_false_view_1], label="sparse", capsize=plot_kwargs['capsize'])

ax[0].errorbar(xticks_no, [x[0] for x in dreamer_test_only_false_view_05], yerr=[x[1] for x in dreamer_test_only_false_view_05], label="Dreamer", capsize=plot_kwargs['capsize'])
ax[1].errorbar(xticks_no, [x[0] for x in dreamer_test_only_false_view_075], yerr=[x[1] for x in dreamer_test_only_false_view_075], label="Dreamer", capsize=plot_kwargs['capsize'])
ax[2].errorbar(xticks_no, [x[0] for x in dreamer_test_only_false_view_1], yerr=[x[1] for x in dreamer_test_only_false_view_1], label="Dreamer", capsize=plot_kwargs['capsize'])

plt.suptitle("Test_only_dropout = False")
ax[0].set_title('p=0.5')
ax[1].set_title('p=0.75')
ax[2].set_title('p=1')
ax[0].set_xlabel(r"Dropout probability $d$")
ax[1].set_xlabel(r"Dropout probability $d$")
ax[2].set_xlabel(r"Dropout probability $d$")
ax[0].set_xticks(xticks_no, xticks_no)
ax[1].set_xticks(xticks_no, xticks_no)
ax[2].set_xticks(xticks_no, xticks_no)

ax[0].set_ylabel("Tuning loss")
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.savefig("test_only_false.pdf")