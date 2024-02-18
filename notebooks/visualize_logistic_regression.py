# %%
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

import os
if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
print(f"Working in {os.getcwd()}")

# config has frame names
print("loading config and paths")
with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

# paths to data files ( need to fix this)
with open("workflow/paths.json", "r") as pf:
    paths = json.loads(pf.read())
print("config and paths loaded")

# load the results dictionary
with open(paths["regression"]["result_pickles"] + "self_influence.pkl", "rb") as self_pkl:
    self_results = pickle.load(self_pkl)

granger_causal_frames = ["Morality and Ethics",
                         "Victim: Humanitarian",
                         "Legality, Constitutionality, Jurisdiction",
                         "Crime and Punishment",
                         "Policy Prescription and Evaluation",
                         "Security and Defense",
                         "External Regulation and Reputation"]

all_frames = config["frames"]["generic"] + config["frames"]["specific"]# + config["frames"]["narrative"]
good_frames = [frame for frame in all_frames if frame not in config["frames"]["low_f1"]]
# %%
frame = "Economic"

print(self_results[frame].pvalues)
print(self_results[frame].params)
# %%
collected_self_results = {"frame": [], "coef": [], "pvalue": [], "ci": []}
for frame in good_frames:
    collected_self_results["frame"].append(frame)
    collected_self_results["coef"].append(self_results[frame].params["exposure"])
    collected_self_results["pvalue"].append(self_results[frame].pvalues["exposure"])
    collected_self_results["ci"].append(self_results[frame].conf_int().loc["exposure"][1] - self_results[frame].params["exposure"])

exposure_results = pd.DataFrame(collected_self_results)
exposure_results["bonferroni_pvalue"] = exposure_results["pvalue"] * exposure_results.shape[0]
exposure_results["Significant"] = exposure_results["bonferroni_pvalue"] < 0.05

# %%
exposure_results = exposure_results.sort_values("coef", ascending=False)

fig, ax = plt.subplots(dpi=300)
sns.barplot(data=exposure_results, x="coef", y="frame",
            hue="Significant", dodge=False, ax=ax)

ax.hlines(range(len(exposure_results)),
          exposure_results["coef"] - exposure_results["ci"],
          exposure_results["coef"] + exposure_results["ci"],
          color="black", lw=3)

ax.set_ylabel('')
plt.tight_layout()
ax.set_xlabel("Self-exposure Log Odds-Ratio")
plt.savefig("plots/regression/self_influence_treatment.png")
plt.savefig("plots/regression/self_influence_treatment.pdf")
plt.show()
# %%
collected_self_results = {"frame": [], "coef": [], "pvalue": [], "ci": []}
for frame in good_frames:
    collected_self_results["frame"].append(frame)
    collected_self_results["coef"].append(self_results[frame].params["ideology"])
    collected_self_results["pvalue"].append(self_results[frame].pvalues["ideology"])

ideology_results = pd.DataFrame(collected_self_results)
ideology_results["bonferroni_pvalue"] = ideology_results["pvalue"] * ideology_results.shape[0]
ideology_results["Significant"] = ideology_results["bonferroni_pvalue"] < 0.05

# %%
ideology_results = ideology_results.sort_values("coef", ascending=False)

fig, ax = plt.subplots(dpi=300)
sns.barplot(data=ideology_results, x="coef", y="frame",
            hue="Significant", dodge=False, ax=ax)

ax.set_ylabel('')
plt.tight_layout()
ax.set_xlabel("Self-exposure Ideology Coefficient")
plt.savefig("plots/regression/self_influence_ideology.png")
plt.savefig("plots/regression/self_influence_ideology.pdf")
plt.show()
# %%
with open(paths["regression"]["result_pickles"] + "alter_influence.pkl", "rb") as self_pkl:
    alter_results = pickle.load(self_pkl)

collected_alter_results = {"frame": [], "coef": [], "pvalue": []}
for frame in good_frames:
    collected_alter_results["frame"].append(frame)
    collected_alter_results["coef"].append(alter_results[frame].params["exposure"])
    collected_alter_results["pvalue"].append(alter_results[frame].pvalues["exposure"])

exposure_results = pd.DataFrame(collected_alter_results)
exposure_results["bonferroni_pvalue"] = exposure_results["pvalue"] * exposure_results.shape[0]
exposure_results["Significant"] = exposure_results["bonferroni_pvalue"] < 0.05

# %%
exposure_results = exposure_results.sort_values("coef", ascending=False)
fig, ax = plt.subplots(dpi=300)
sns.barplot(data=exposure_results, x="coef", y="frame",
            hue="Significant", dodge=False, ax=ax)
ax.set_ylabel('')
plt.tight_layout()
ax.set_xlabel("Alter-exposure Log Odds-Ratio")
plt.savefig("plots/regression/alter_influence_treatment.png")
plt.savefig("plots/regression/alter_influence_treatment.pdf")
plt.show()

# %%
