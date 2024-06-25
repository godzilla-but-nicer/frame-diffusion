# %% [markdown]
#
# # Loading data etc
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

# wse want to use only a subset of the frames
granger_causal_frames = ["Morality and Ethics",
                         "Victim: Humanitarian",
                         "Legality, Constitutionality, Jurisdiction",
                         "Crime and Punishment",
                         "Policy Prescription and Evaluation",
                         "Security and Defense",
                         "External Regulation and Reputation"]

# lets just drop the poor performers and the narrative frames
all_frames = config["frames"]["generic"] + config["frames"]["specific"]# + config["frames"]["narrative"]
good_frames = [frame for frame in all_frames if frame not in config["frames"]["low_f1"]]
# %% [markdown]
#
# ## Self-Exposure Log-Odds Ratios
#
# First we build a dataframe with all of the relevant values to make the plot
#  this isw basically the frames and their associated numbers. Then we make the
#  plot.
#
# This one shows the log-odds ratio for cueing frame x in a particular tweet.
# Specifically, this means the odds of cuing the frame given the user cued the
# frame the previous day over the odds of cuing the frame given the user did
# not cue the frame the previous day.
#
# This value is on the x-axis along with confidence intervals
# %%
collected_self_results = {"frame": [], "coef": [], "pvalue": [], "ci": [], "Frame Type": []}

for frame in good_frames:

    collected_self_results["frame"].append(frame)
    collected_self_results["coef"].append(self_results[frame].params["exposure"])
    collected_self_results["pvalue"].append(self_results[frame].pvalues["exposure"])
    collected_self_results["ci"].append(self_results[frame].conf_int().loc["exposure"][1] - self_results[frame].params["exposure"])
    
    if frame in config["frames"]["generic"]:
        ftype = "Issue-Generic"
    if frame in config["frames"]["specific"]:
        ftype = "Issue-Specific"
    collected_self_results["Frame Type"].append(ftype)

exposure_results = pd.DataFrame(collected_self_results)
exposure_results["bonferroni_pvalue"] = exposure_results["pvalue"] * exposure_results.shape[0]
exposure_results["Significant"] = exposure_results["bonferroni_pvalue"] < 0.05

exposure_results = exposure_results.sort_values("coef", ascending=False)


# Now we're starting to build the self-exposure figure
fig, ax = plt.subplots(dpi=300)

# bars
sns.barplot(data=exposure_results, x="coef", y="frame",
            hue="Frame Type", dodge=False, ax=ax)

# error bars
ax.hlines(range(len(exposure_results)),
          exposure_results["coef"] - exposure_results["ci"],
          exposure_results["coef"] + exposure_results["ci"],
          color="black", lw=2)

# remove some of the borders for prettiness
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# tighten up the vertical space
ax.set_ylim((len(exposure_results)-0.4, -0.6))

ax.set_ylabel('')
plt.tight_layout()
ax.set_xlabel("Self-exposure Log Odds-Ratio")
plt.savefig("plots/regression/self_influence_treatment.png")
plt.savefig("plots/regression/self_influence_treatment.pdf")
plt.show()
# %% [markdown]
#
# ## Relationship between frame and ideology accounting for self-exposure
#
# Here we're going to put ideology on the x-axis. More positive means more
# right wing users use the frame negative means left-wing.
# %%
collected_self_results = {"frame": [], "coef": [], "pvalue": [], "ci": [], "Frame Type": []}

for frame in good_frames:
    collected_self_results["frame"].append(frame)
    collected_self_results["coef"].append(self_results[frame].params["ideology"])
    collected_self_results["pvalue"].append(self_results[frame].pvalues["ideology"])
    collected_self_results["ci"].append(self_results[frame].conf_int().loc["ideology"][1] - self_results[frame].params["ideology"])

    if frame in config["frames"]["generic"]:
        ftype = "Generic"
    if frame in config["frames"]["specific"]:
        ftype = "Specific"

    collected_self_results["Frame Type"].append(ftype)

ideology_results = pd.DataFrame(collected_self_results)
ideology_results["bonferroni_pvalue"] = ideology_results["pvalue"] * ideology_results.shape[0]
ideology_results["Significant"] = ideology_results["bonferroni_pvalue"] < 0.05

# %%
ideology_results = ideology_results.sort_values("coef", ascending=False)
ideology_results = ideology_results[ideology_results["Significant"]]

# bars
fig, ax = plt.subplots(dpi=300)
sns.barplot(data=ideology_results, x="coef", y="frame",
            hue="Frame Type", dodge=False, ax=ax)

# error bars
ax.hlines(range(len(ideology_results)),
          ideology_results["coef"] - ideology_results["ci"],
          ideology_results["coef"] + ideology_results["ci"],
          color="black", lw=2)

# tighten up the vertical space
ax.set_ylim((len(ideology_results)-0.4, -0.6))

ax.set_ylabel('')
plt.tight_layout()
ax.set_xlabel("Self-exposure Ideology Coefficient")
plt.savefig("plots/regression/self_influence_ideology.png")
plt.savefig("plots/regression/self_influence_ideology.pdf")
plt.show()
# %% [markdown]
#
# ## Alter-influence figure
#
# This figure shows the log odds ratio for frame cueing but with frame exposure
# from ones mention network neighbors rather than ones self
# %%
with open(paths["regression"]["result_pickles"] + "alter_influence.pkl", "rb") as self_pkl:
    alter_results = pickle.load(self_pkl)

collected_alter_results = {"frame": [], "coef": [], "pvalue": [], "ci": [], "Frame Type": []}

for frame in good_frames:
    collected_alter_results["frame"].append(frame)
    collected_alter_results["coef"].append(alter_results[frame].params["exposure"])
    collected_alter_results["pvalue"].append(alter_results[frame].pvalues["exposure"])
    collected_alter_results["ci"].append(alter_results[frame].conf_int().loc["exposure"][1] - alter_results[frame].params["exposure"])

    if frame in config["frames"]["generic"]:
        ftype = "Generic"
    if frame in config["frames"]["specific"]:
        ftype = "Specific"
    
    collected_alter_results["Frame Type"].append(ftype)

exposure_results = pd.DataFrame(collected_alter_results)
exposure_results["bonferroni_pvalue"] = exposure_results["pvalue"] * exposure_results.shape[0]
exposure_results["Significant"] = exposure_results["bonferroni_pvalue"] < 0.05

exposure_results = exposure_results.sort_values("coef", ascending=False)

significant_results = exposure_results[exposure_results["Significant"]]

fig, ax = plt.subplots(dpi=300)

# bar
sns.barplot(data=significant_results, x="coef", y="frame",
            hue="Frame Type", dodge=False, ax=ax)

ax.legend(title="Frame Type", loc="lower right")

# errorbars
ax.hlines(range(len(significant_results)),
          significant_results["coef"] - significant_results["ci"],
          significant_results["coef"] + significant_results["ci"],
          color="black", lw=2)

# tighten up the vertical space
ax.set_ylim((len(significant_results)-0.4, -0.6))


# remove some of the borders for prettiness
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_ylabel('')
plt.tight_layout()
ax.set_xlabel("Network-exposure Log Odds-Ratio")
plt.savefig("plots/regression/alter_influence_treatment.png")
plt.savefig("plots/regression/alter_influence_treatment.pdf")
plt.show()

# %%
# %% [markdown]
#
# ## Relationship between frame and ideology accounting for alter-exposure
#
# Here we're going to put ideology on the x-axis. More positive means more
# right wing users use the frame negative means left-wing.
# %%
collected_alter_results = {"frame": [], "coef": [], "pvalue": [], "ci": [], "Frame Type": []}

for frame in good_frames:
    collected_alter_results["frame"].append(frame)
    collected_alter_results["coef"].append(alter_results[frame].params["ideology"])
    collected_alter_results["pvalue"].append(alter_results[frame].pvalues["ideology"])
    collected_alter_results["ci"].append(alter_results[frame].conf_int().loc["ideology"][1] - alter_results[frame].params["ideology"])

    if frame in config["frames"]["generic"]:
        ftype = "Generic"
    if frame in config["frames"]["specific"]:
        ftype = "Specific"

    collected_alter_results["Frame Type"].append(ftype)

ideology_results = pd.DataFrame(collected_alter_results)
ideology_results["bonferroni_pvalue"] = ideology_results["pvalue"] * ideology_results.shape[0]
ideology_results["Significant"] = ideology_results["bonferroni_pvalue"] < 0.05

ideology_results = ideology_results.sort_values("coef", ascending=False)
ideology_results = ideology_results[ideology_results["Significant"]]

# bars
fig, ax = plt.subplots(dpi=300)
sns.barplot(data=ideology_results, x="coef", y="frame",
            hue="Frame Type", dodge=False, ax=ax)

# error bars
ax.hlines(range(len(ideology_results)),
          ideology_results["coef"] - ideology_results["ci"],
          ideology_results["coef"] + ideology_results["ci"],
          color="black", lw=2)

# tighten up the vertical space
ax.set_ylim((len(ideology_results)-0.4, -0.6))

ax.set_ylabel('')
plt.tight_layout()
ax.set_xlabel("Network exposure Ideology Coefficient")
plt.savefig("plots/regression/alter_influence_ideology.png")
plt.savefig("plots/regression/alter_influence_ideology.pdf")
plt.show()
# %% [markdown]
# #quick sidebar: correlarion between frequency and log odds ratios
# %%
collected_results = {"frame": [], "exposure": [], "ideology": [], "Frame Type": []}
for frame in good_frames:
    if (alter_results[frame].pvalues["ideology"] * len(good_frames) < 0.05) and \
       (alter_results[frame].pvalues["exposure"] * len(good_frames) < 0.05):

        collected_results["frame"].append(frame)
        collected_results["ideology"].append(alter_results[frame].params["ideology"])
        collected_results["exposure"].append(alter_results[frame].params["exposure"])

        if frame in config["frames"]["generic"]:
            ftype = "Generic"
        if frame in config["frames"]["specific"]:
            ftype = "Specific"

        collected_results["Frame Type"].append(ftype)

collected_df = pd.DataFrame(collected_results)
# %%
frame_probs = pd.read_csv("data/eda_bootstrap/coarse_frequency_boot.tsv", sep="\t", index_col=0)
public_probs = frame_probs[frame_probs["Group"] == "public"]
good_probs = public_probs[public_probs["Frame"].isin(good_frames)]
good_probs = good_probs[["Frame", "Frequency"]].rename({"Frame": "frame"}, axis="columns")
odds_and_probs = pd.merge(collected_df, good_probs, on="frame")

# calculate correlation after plotting though wow
plt.figure(figsize=(7,5))
sns.scatterplot(odds_and_probs, x="Frequency", y="exposure", s=100, hue="Frame Type")
for i, frame in enumerate(odds_and_probs["frame"]):
    plt.text(odds_and_probs["Frequency"][i] + 0.01, odds_and_probs["exposure"][i], frame, fontsize="small")

plt.ylabel("Network-Exposure Log Odds-Ratio")
plt.xlabel("Frame Usage Frequency")
plt.tight_layout()
plt.savefig("plots/regression/alter_odds_frequency.svg", dpi=300)

# ok so its going to be negativly correlated and spearman looks appropriate
from scipy.stats import spearmanr

# %%
collected_self_results = {"frame": [], "coef": [], "pvalue": [], "ci": [], "Frame Type": []}

for frame in good_frames:

    collected_self_results["frame"].append(frame)
    collected_self_results["coef"].append(self_results[frame].params["exposure"])
    collected_self_results["pvalue"].append(self_results[frame].pvalues["exposure"])
    collected_self_results["ci"].append(self_results[frame].conf_int().loc["exposure"][1] - self_results[frame].params["exposure"])
    
    if frame in config["frames"]["generic"]:
        ftype = "Generic"
    if frame in config["frames"]["specific"]:
        ftype = "Specific"
    collected_self_results["Frame Type"].append(ftype)

exposure_results = pd.DataFrame(collected_self_results)
exposure_results["bonferroni_pvalue"] = exposure_results["pvalue"] * exposure_results.shape[0]
exposure_results["Significant"] = exposure_results["bonferroni_pvalue"] < 0.05

self_df = exposure_results.copy()

self_df["self_coef"] = self_df["coef"]
self_df["self_p"] = self_df["bonferroni_pvalue"]


collected_alter_results = {"frame": [], "coef": [], "pvalue": [], "ci": [], "Frame Type": []}

for frame in good_frames:
    collected_alter_results["frame"].append(frame)
    collected_alter_results["coef"].append(alter_results[frame].params["exposure"])
    collected_alter_results["pvalue"].append(alter_results[frame].pvalues["exposure"])
    collected_alter_results["ci"].append(alter_results[frame].conf_int().loc["exposure"][1] - alter_results[frame].params["exposure"])

    if frame in config["frames"]["generic"]:
        ftype = "Generic"
    if frame in config["frames"]["specific"]:
        ftype = "Specific"
    
    collected_alter_results["Frame Type"].append(ftype)

exposure_results = pd.DataFrame(collected_alter_results)
exposure_results["bonferroni_pvalue"] = exposure_results["pvalue"] * exposure_results.shape[0]
exposure_results["Significant"] = exposure_results["bonferroni_pvalue"] < 0.05

alter_df = exposure_results.copy()

alter_df["alter_coef"] = alter_df["coef"]
alter_df["alter_p"] = alter_df["bonferroni_pvalue"]

combined = pd.merge(self_df[["frame", "Frame Type", "self_coef", "self_p"]],
                    alter_df[["frame", "Frame Type", "alter_coef", "alter_p"]],
                    on=["frame", "Frame Type"])

def max_p_val(row):
    return max([row["self_p"], row["alter_p"]])

combined["big_p"] = combined.apply(max_p_val, axis=1)
# %%
fig, ax = plt.subplots(figsize=(7,5))
sns.scatterplot(combined, x="self_coef", y="alter_coef", hue="Frame Type", ax=ax)
ax.set_xlabel(r"Self-Exposure $\beta$")
ax.set_ylabel(r"Alter-Exposure $\beta$")
plt.tight_layout()
plt.savefig("plots/regression/alter_and_self_odds.png", dpi=300)

print(spearmanr(combined["self_coef"], combined["alter_coef"]))
# %%
