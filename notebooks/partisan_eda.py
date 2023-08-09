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

from frame_stats import bootstrap_ci, draw_frame_frequencies
import data_selector as ds


with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

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
                                    10000,
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
figg.savefig("../plots/political_eda/congress_generic.pdf")
figg.savefig("../plots/political_eda/congress_generic.png")
plt.show()
# %%
figs, axs = plt.subplots(figsize=(7.2, 6.4))

draw_frame_frequencies(frame_probs, config["frames"]["specific"], axs,
                       group_col="Affiliation",
                       estimate_col="estimate",
                       colors=["blue", "red", "green", "orange"])

plt.tight_layout()
figs.savefig("../plots/political_eda/congress_specific.pdf")
figs.savefig("../plots/political_eda/congress_specific.png")
plt.show()
# %%
fign, axn = plt.subplots()

draw_frame_frequencies(frame_probs, config["frames"]["narrative"], axn,
                       group_col="Affiliation",
                       estimate_col="estimate",
                       colors=["blue", "red", "green", "orange"],
                       legend_loc="upper left")

plt.tight_layout()
fign.savefig("../plots/political_eda/congress_narrative.pdf")
fign.savefig("../plots/political_eda/congress_narrative.png")
plt.show()
# %%
fig, ax = plt.subplots(ncols=2, figsize=(7.2, 4.8))

generic_frames = config["frames"]["generic"]
generic_sums = frame_sums[generic_frames]

# renormalize
normed_cols = {}
for i, frame in enumerate(generic_sums.columns):
    normed_cols[frame] = []
    for i, row in generic_sums.iterrows():
        normed_cols[frame].append(row[frame] / generic_sums.iloc[i].sum())

generic_normed = pd.DataFrame(normed_cols)
generic_normed["Affiliation"] = frame_sums["Affiliation"]

generic_long = pd.melt(generic_normed, id_vars="Affiliation")

sns.barplot(generic_long, x="value", y="variable", hue="Affiliation", ax=ax[0])


specific_frames = config["frames"]["specific"]
specific_sums = frame_sums[specific_frames]

normed_cols = {}
for i, frame in enumerate(specific_sums.columns):
    normed_cols[frame] = []
    for i, row in specific_sums.iterrows():
        normed_cols[frame].append(row[frame] / specific_sums.iloc[i].sum())

specific_normed = pd.DataFrame(normed_cols)
specific_normed["Affiliation"] = frame_sums["Affiliation"]

specific_long = pd.melt(specific_normed, id_vars="Affiliation")

sns.barplot(specific_long, x="value", y="variable",
            hue="Affiliation", ax=ax[1])

plt.tight_layout()
plt.show()
# %% [markdown]
#
# Let's get frame diversity now
#

# %%
fig, ax = plt.subplots(nrows=2, figsize=(5, 4))

gen_divs = []
for i, row in generic_normed.iterrows():
    print(row[:-1].values)
    gen_divs.append(
        entropy(row[:-1].astype(float).values) / np.log(row[:-1].shape[0]))

gen_div_df = pd.DataFrame({"Affiliation": generic_normed["Affiliation"].values,
                          "Frame Diversity": gen_divs})

spe_divs = []
for i, row in specific_normed.iterrows():
    spe_divs.append(
        entropy(row[:-1].astype(float).values) / np.log(row[:-1].shape[0]))

spe_div_df = pd.DataFrame({"Affiliation": specific_normed["Affiliation"].values,
                          "Frame Diversity": spe_divs})

sns.barplot(gen_div_df, x="Affiliation", y="Frame Diversity", ax=ax[0])
sns.barplot(spe_div_df, x="Affiliation", y="Frame Diversity", ax=ax[1])
plt.tight_layout()
plt.show()
# %%
