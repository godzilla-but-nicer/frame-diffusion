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
# %%
frame = "Economic"

print(self_results[frame].pvalues)
print(self_results[frame].params)
# %%
collected_self_results = {"frame": [], "coef": [], "pvalue": []}
for frame in self_results.keys():
    collected_self_results["frame"].append(frame)
    collected_self_results["coef"].append(self_results[frame].params["exposure"])
    collected_self_results["pvalue"].append(self_results[frame].pvalues["exposure"])

exposure_results = pd.DataFrame(collected_self_results)
exposure_results["bonferroni_pvalue"] = exposure_results["pvalue"] * exposure_results.shape[0]

bad_prediction = []
for _, row in exposure_results.iterrows():
    if row["frame"] in config["frames"]["low_f1"]:
        bad_prediction.append(True)
    else:
        bad_prediction.append(False)

exposure_results["bad_f1"] = bad_prediction


# %%
exposure_results = exposure_results.sort_values("coef", ascending=False)
fig, ax = plt.subplots(dpi=300)
sns.barplot(data=exposure_results, x="coef", y="frame",
            hue="bad_f1", dodge=False, ax=ax)
ax.set_ylabel('')
plt.tight_layout()
ax.set_xlabel("Self-exposure Log Odds-Ratio")
plt.savefig("plots/regression/self_influence_treatment.png")
plt.savefig("plots/regression/self_influence_treatment.pdf")
plt.show()
# %%
from scipy.stats import ranksums

specific = []
generic = []
for _, row in exposure_results.iterrows():
    if ":" in row["frame"] and row["frame"] not in config["frames"]["low_f1"]:
        specific.append(row["coef"])
    elif ":" not in row["frame"] and row["frame"] not in config["frames"]["low_f1"]:
        generic.append(row["coef"])

print(ranksums(generic, specific))
# %%
collected_self_results = {"frame": [], "coef": [], "pvalue": []}
for frame in self_results.keys():
    collected_self_results["frame"].append(frame)
    collected_self_results["coef"].append(self_results[frame].params["ideology"])
    collected_self_results["pvalue"].append(self_results[frame].pvalues["ideology"])

ideology_results = pd.DataFrame(collected_self_results)
ideology_results["bonferroni_pvalue"] = ideology_results["pvalue"] * ideology_results.shape[0]

bad_prediction = []
for _, row in ideology_results.iterrows():
    if row["frame"] in config["frames"]["low_f1"]:
        bad_prediction.append(True)
    else:
        bad_prediction.append(False)

ideology_results["bad_f1"] = bad_prediction

# %%
ideology_results = ideology_results.sort_values("coef", ascending=False)
fig, ax = plt.subplots(dpi=300)
sns.barplot(data=ideology_results, x="coef", y="frame",
            hue="bad_f1", dodge=False, ax=ax)
ax.set_ylabel('')
plt.tight_layout()
ax.set_xlabel("Self-exposure Ideology Coefficient")
plt.savefig("plots/regression/self_influence_ideology.png")
plt.savefig("plots/regression/self_influence_ideology.pdf")
plt.show()
# %%
