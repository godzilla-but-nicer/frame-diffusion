# %% [markdown]
# # Political Frame EDA
#
# Ok here we're going to basically do the same stuff as in the frame EDA
# notebook but this time specifically looking at subsets of the congress tweets
# corresponding to party and subsets of the journalists corresponding to
# political ideology.
#
# We'll start by loading a whole bunch of data and then we'll get into looking
# at frame distributions.
#
# %%
import os
if not os.getcwd().split("/")[-1] == "frame-diffusion":
    os.chdir("..")

import json as json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from functools import reduce
from scipy.stats import entropy
from typing import Dict

from frame_stats import bootstrap_ci, draw_frame_frequencies
import data_selector as ds


with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

with open("workflow/paths.json", "r") as cf:
    paths = json.loads(cf.read())
# %%
congress_preds = ds.load_congress_frames()
# %%
frame_cols = config["frames"]["generic"] + config["frames"]["specific"] + config["frames"]["narrative"]
new_rows = []
for i, frame in enumerate(congress_preds[frame_cols]):
    for affiliation in congress_preds["Affiliation"].unique():
        new_row = {"Affiliation": affiliation}
        new_row["Frame"] = frame

        aff_preds = congress_preds[congress_preds["Affiliation"]
                                   == affiliation]
        new_row.update(bootstrap_ci(aff_preds[frame].values,
                                    lambda x: x.sum() / x.shape[0],
                                    1000,
                                    0.05))

        new_rows.append(new_row)

frame_probs = pd.DataFrame(new_rows)
# %% [markdown]
# ## Frame Frequency
#
# We're goiong to look at frame frequencies for all of these groups like we did
# with the coarser groups.
#

# %%
figg, axg = plt.subplots(figsize=(7.2, 6.4))

draw_frame_frequencies(frame_probs, config["frames"]["generic"], axg,
                       group_col="Affiliation",
                       estimate_col="estimate",
                       colors=["blue", "red", "green", "orange"])

plt.tight_layout()
figg.savefig("plots/political_eda/congress_generic.pdf")
figg.savefig("plots/political_eda/congress_generic.png")
plt.show()
# %%
figs, axs = plt.subplots(figsize=(7.2, 6.4))

draw_frame_frequencies(frame_probs, config["frames"]["specific"], axs,
                       group_col="Affiliation",
                       estimate_col="estimate",
                       colors=["blue", "red", "green", "orange"])

plt.tight_layout()
figs.savefig("plots/political_eda/congress_specific.pdf")
figs.savefig("plots/political_eda/congress_specific.png")
plt.show()
# %%
fign, axn = plt.subplots()

draw_frame_frequencies(frame_probs, config["frames"]["narrative"], axn,
                       group_col="Affiliation",
                       estimate_col="estimate",
                       colors=["blue", "red", "green", "orange"],
                       legend_loc="upper left")

plt.tight_layout()
fign.savefig("plots/political_eda/congress_narrative.pdf")
fign.savefig("plots/political_eda/congress_narrative.png")
plt.show()

# %% [markdown]
# # Let's do frame diversity as measured by entropy. Probably more interesting with more categories
# %%
def renormalize_estimates(long_frame_probs: pd.DataFrame,
                          group_col: str,
                          group_label: str,
                          prob_col: str = "estimate",
                          frame_col: str = "Frame",
                          config: Dict = config) -> pd.DataFrame:

    group_probs = long_frame_probs[long_frame_probs[group_col] == group_label]
    group_not_narrative = group_probs[~group_probs[frame_col].isin(config["frames"]["narrative"])].copy()
    normed_probabilities = group_not_narrative[prob_col].values / group_not_narrative[prob_col].values.sum()
    group_not_narrative[prob_col + "_normed"] = normed_probabilities
    return group_not_narrative

rep_normed = renormalize_estimates(frame_probs, "Affiliation", "Republican")
dem_normed = renormalize_estimates(frame_probs, "Affiliation", "Democrat")

rep_entropy = entropy(rep_normed["estimate_normed"]) / np.log(rep_normed.shape[0])
dem_entropy = entropy(dem_normed["estimate_normed"]) / np.log(dem_normed.shape[0])


# %%


journo = ds.load_journalist_frames()
# %% [markdown]
# ## we need to figure out the distribution of our journalist biases

# ok

# %%
plt.hist(journo["bias_rating"], bins=20)
# %% [markdown]
# Its super imbalanced but I think actually I dont care. I think we should just
# use 0 as our threshold ant way because it nominally has real world significance

# %%
def bias_to_label(bias: float):
    if bias < 0:
        return "liberal"
    if bias > 0:
        return "conservative"
    else:
        return np.nan

journo["affiliation"] = journo["bias_rating"].map(bias_to_label)
journo = journo.dropna(subset="affiliation")


new_rows = []
for i, frame in enumerate(journo[frame_cols]):
    for affiliation in journo["affiliation"].unique():
        new_row = {"Affiliation": affiliation}
        new_row["Frame"] = frame

        journo_aff = journo[journo["affiliation"]
                                   == affiliation]
        new_row.update(bootstrap_ci(journo_aff[frame].values,
                                    lambda x: x.sum() / x.shape[0],
                                    1000,
                                    0.05))

        new_rows.append(new_row)

journo_frame_probs = pd.DataFrame(new_rows)

# %%
fig, ax = plt.subplots()
draw_frame_frequencies(journo_frame_probs, plot_frames=config["frames"]["generic"],
                       ax=ax, group_col="Affiliation", estimate_col="estimate")
# %%
fig, ax = plt.subplots()
draw_frame_frequencies(journo_frame_probs, plot_frames=config["frames"]["specific"],
                       ax=ax, group_col="Affiliation", estimate_col="estimate")
plt.tight_layout()
fig.savefig("plots/political_eda/journo_specific.png", dpi=300)
# %%
# The Public
# %%
public_frames = ds.load_public_frames()

# %% [markdown]
# Want to get ideology distribution for members of the public to start
# %%
public_frames = public_frames[~public_frames["ideology"].isna()]
mean_ideology = np.mean(public_frames["ideology"])
plt.hist(public_frames["ideology"], bins=40)
plt.axvline(mean_ideology)
# %% [markdown]
# I'll start naive and just use zero as the threshold again

# %%
public_frames["affiliation"] = public_frames["ideology"].map(bias_to_label)

new_rows = []
for i, frame in enumerate(public_frames[frame_cols]):
    for affiliation in public_frames["affiliation"].unique():
        new_row = {"Affiliation": affiliation}
        new_row["Frame"] = frame

        public_aff = public_frames[public_frames["affiliation"]
                                   == affiliation]
        new_row.update(bootstrap_ci(public_aff[frame].values,
                                    lambda x: x.sum() / x.shape[0],
                                    100,
                                    0.05))

        new_rows.append(new_row)

public_frame_probs = pd.DataFrame(new_rows)
# %%
fig, ax = plt.subplots()
draw_frame_frequencies(public_frame_probs, plot_frames=config["frames"]["generic"],
                       ax=ax, group_col="Affiliation", estimate_col="estimate")
# %%
fig, ax = plt.subplots()
draw_frame_frequencies(public_frame_probs, plot_frames=config["frames"]["specific"],
                       ax=ax, group_col="Affiliation", estimate_col="estimate")
fig.savefig("plots/political_eda/public_specific.png")

# %%
# Frame diversity time finally, baby

# %%
lib_journo_normed = renormalize_estimates(journo_frame_probs, "Affiliation", "liberal")
con_journo_normed = renormalize_estimates(journo_frame_probs, "Affiliation", "conservative")
lib_public_normed = renormalize_estimates(public_frame_probs, "Affiliation", "liberal")
con_public_normed = renormalize_estimates(public_frame_probs, "Affiliation", "conservative")

# %%
from frame_stats.frame_stats import residual_ci, percentile_ci
def entropy_ci(label, affiliation, frequencies, samples, alpha):

    sample_estimates = []
    for _ in range(samples):
        sample = np.random.choice(frequencies, size=frequencies.shape[0], replace=True)
        sample_estimates.append(entropy(sample) / np.log(frequencies.shape[0]))
    
    point_estimate = entropy(frequencies) / np.log(frequencies.shape[0])

    lower, upper = residual_ci(sample_estimates, alpha, point_estimate)

    return {"Group": label, "Affiliation": affiliation, "estimate": point_estimate, "lower": lower, "upper": upper}

lib_journo_entropy = entropy_ci("Journalists", "Left", lib_journo_normed["estimate_normed"], 100, 0.05)
con_journo_entropy = entropy_ci("Journalists", "Right", con_journo_normed["estimate_normed"], 100, 0.05)
lib_public_entropy = entropy_ci("Public", "Left", lib_public_normed["estimate_normed"], 100, 0.05)
con_public_entropy = entropy_ci("Public", "Right", con_public_normed["estimate_normed"], 100, 0.05)
dem_entropy = entropy_ci("Congress", "Left", dem_normed["estimate_normed"], 100, 0.05)
rep_entropy = entropy_ci("Congress", "Right", rep_normed["estimate_normed"], 100, 0.05)

diversity_df = pd.DataFrame([lib_journo_entropy, con_journo_entropy, lib_public_entropy, con_public_entropy, dem_entropy, rep_entropy])

fig, ax = plt.subplots()
sns.barplot(diversity_df, x="estimate", y="Group", hue="Affiliation", ax=ax)
# %%
# %%
