# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json as json
from functools import reduce
from scipy.stats import entropy
from frame_stats import bootstrap_ci, bootstrap_ci_multivariate, draw_frame_frequencies
from itertools import product

import os
if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
print(f"Working in {os.getcwd()}")

with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

# load data
groups = ["congress", "journalists", "trump", "public"]
frame_types = ["generic", "specific", "narrative"]
all_frames = config["frames"]["specific"].copy()
all_frames.extend(config["frames"]["generic"].copy())
all_frames.extend(config["frames"]["narrative"].copy())

# %%
rerun_bootstrap = False

if rerun_bootstrap:
    frame_probs = {k: {} for k in groups} # will hold a bunch of vectors of frame sums by group

    for group in groups:

        if group != "public":
            group_dfs = []
            for frame_type in frame_types:
                predictions = pd.read_csv(f"../data/binary_frames/{group}/{group}_{frame_type}.tsv",
                                          sep="\t")
                group_dfs.append(predictions.drop("text", axis="columns")) 
            predictions = (reduce(lambda l, r: pd.merge(l, r, on="id_str"),
                              group_dfs).fillna(0))
        else:
            # list of data frames for the group to combine into one
                predictions = pd.read_csv(f"../data/binary_frames/predicted_frames.tsv",
                                          sep="\t")
                us_public_ids = pd.read_csv(f"../data/us_public_ids.tsv", sep="\t")
                predictions = pd.merge(us_public_ids, predictions,
                                       on="id_str", how="left")
                group_dfs.append(predictions)


        group_frames = {}
        for i, frame in enumerate(all_frames):

            group_frames[frame] = bootstrap_ci(predictions[frame].values,
                                               lambda x: x.sum() / x.shape[0],
                                               10000,
                                               0.05,
                                               seed=123)

        frame_probs[group] = group_frames


        long_ids = product(groups, all_frames)
        long_probs = pd.DataFrame(long_ids, columns=["Group", "Frame"])
        lowers = []
        estimates = []
        uppers = []

        for i, row in long_probs.iterrows():
            lowers.append(frame_probs[row["Group"]][row["Frame"]]["lower"])
            estimates.append(frame_probs[row["Group"]][row["Frame"]]["estimate"])
            uppers.append(frame_probs[row["Group"]][row["Frame"]]["upper"])

        long_probs["lower"] = lowers
        long_probs["Frequency"] = estimates
        long_probs["upper"] = uppers

        generic_long = long_probs[long_probs["Frame"].isin(config["frames"]["generic"])]
        specific_long = long_probs[long_probs["Frame"].isin(config["frames"]["specific"])]
        narrative_long = long_probs[long_probs["Frame"].isin(config["frames"]["narrative"])]

else:

    long_probs = pd.read_csv("../data/eda_bootstrap/coarse_frequency_boot.tsv", sep="\t")

generic_long = long_probs[long_probs["Frame"].isin(config["frames"]["generic"])]
specific_long = long_probs[long_probs["Frame"].isin(config["frames"]["specific"])]
narrative_long = long_probs[long_probs["Frame"].isin(config["frames"]["narrative"])]
# %%
figg, axg = plt.subplots()

draw_frame_frequencies(long_probs, config["frames"]["generic"], axg)

figg.tight_layout()
figg.savefig(f"../plots/frame_eda/generic_frequency_by_group.png")
figg.savefig(f"../plots/frame_eda/generic_frequency_by_group.pdf")

# %%
figs, axs = plt.subplots()

draw_frame_frequencies(long_probs, config["frames"]["specific"], axs)

figs.tight_layout()
figs.savefig(f"../plots/frame_eda/specific_frequency_by_group.png")
figs.savefig(f"../plots/frame_eda/specific_frequency_by_group.pdf")

# %%
fign, axn = plt.subplots()

draw_frame_frequencies(long_probs, config["frames"]["narrative"], axn, 
                       legend_loc="upper left")

fign.tight_layout()
fign.savefig(f"../plots/frame_eda/narrative_frequency_by_group.png")
fign.savefig(f"../plots/frame_eda/narrative_frequency_by_group.pdf")

# %%
frames = {}
for group in groups:

    if group != "public":
        group_dfs = []
        for frame_type in frame_types:
            predictions = pd.read_csv(f"../data/binary_frames/{group}/{group}_{frame_type}.tsv",
                                      sep="\t")
            group_dfs.append(predictions.drop("text", axis="columns"))

        predictions = (reduce(lambda l, r: pd.merge(l, r, on="id_str"),
                          group_dfs).fillna(0))
    else:
        # list of data frames for the group to combine into one
            predictions = pd.read_csv(f"../data/binary_frames/predicted_frames.tsv",
                                      sep="\t")
            us_public_ids = pd.read_csv(f"../data/us_public_ids.tsv", sep="\t")
            predictions = pd.merge(us_public_ids, predictions,
                                   on="id_str", how="left")
            group_dfs.append(predictions)
    
    frames[group] = predictions[all_frames]

# %%

def diversity(df):
    raw = df.values
    frame_distribution = np.sum(raw, axis=0) / np.sum(raw)

    normed_entropy = entropy(frame_distribution) / np.log(frame_distribution.shape[0])

    return normed_entropy

# %%
rerun_bootstrap = False

if rerun_bootstrap:
    diversity_rows = []
    for frame_type in frame_types:
        type_rows = []
        for group in groups:

            gtdf = frames[group][config["frames"][frame_type]]

            diversity_ci = bootstrap_ci_multivariate(gtdf,
                                                     diversity,
                                                     10000,
                                                     "rows",
                                                     0.05)

            row_dict = {"group": group,
                        "frame_type": frame_type,
                        "lower": diversity_ci["lower"],
                        "diversity": diversity_ci["estimate"],
                        "upper": diversity_ci["upper"]}
            diversity_rows.append(row_dict)


    div_df = pd.DataFrame(diversity_rows)

else:
    div_df = pd.read_csv("../data/eda_bootstrap/coarse_diversity_boot.tsv", sep="\t")
# %%
fig, ax = plt.subplots()

pad = 2

bar_x_coords = [np.arange(i, 
                          (len(groups) + pad) * len(frame_types) + i,
                          len(groups) + pad)
                for i in range(len(groups))]

xticks = np.arange(1.5,
                   (len(groups)+pad) * len(frame_types),
                   len(groups) + pad)
for i, group in enumerate(groups):
    group_df = div_df[div_df["group"] == group]

    ax.scatter(bar_x_coords[i], group_df["diversity"], label=f"{group.title()}")
    ax.vlines(bar_x_coords[i], group_df["lower"], group_df["upper"], color="black")

ax.set_xticks(xticks)
ax.set_xticklabels([ft.title() for ft in frame_types])
ax.set_ylabel("Diversity")
ax.legend(loc="upper left")

fig.savefig("../plots/frame_eda/coarse_diversity.png")
fig.savefig("../plots/frame_eda/coarse_diversity.pdf")
# %%
frame_matrices = {}
frame_labels = {k: [] for k in frame_types}

for frame_type in frame_types:
    type_mats = []
    for group in groups:

        # all frames are in one file for the public
        if group != "public":
            predictions = pd.read_csv(f"../data/binary_frames/{group}/{group}_{frame_type}.tsv",
                                      sep="\t")
        else:
            predictions = pd.read_csv(f"../data/binary_frames/predicted_frames.tsv",
                                      sep="\t")
            us_public_ids = pd.read_csv(f"../data/us_public_ids.tsv", sep="\t")
            predictions = pd.merge(us_public_ids, predictions,
                                   on="id_str", how="left")
            
        # add labels to the label dict
        frame_labels[frame_type].extend([group]*predictions.shape[0])
        
        # we essentially want a feature array for each group
        group_mat = np.zeros((predictions.shape[0], len(config["frames"][frame_type])))
        for i, frame in enumerate(config["frames"][frame_type]):

            group_mat[:, i] = predictions[frame].values
        
        type_mats.append(group_mat)
        
    # put everything in the matrix dict
    frame_matrices[frame_type] = np.vstack(type_mats)

    frame_labels[frame_type] = np.array(frame_labels[frame_type])

# %%
# overall correlations
corr_generic = np.corrcoef(frame_matrices["generic"].T)
corr_specific = np.corrcoef(frame_matrices["specific"].T)
corr_narrative = np.corrcoef(frame_matrices["narrative"].T)

def draw_heatmap(corrmat, labels):
    cmap = sns.diverging_palette(20, 230, as_cmap=True)
    sns.heatmap(corrmat, annot=True, fmt=".1f", center=0, cmap=cmap,
                vmin=-1, vmax=1, square=True,
                xticklabels=labels,
                yticklabels=labels)

draw_heatmap(corr_generic, config["frames"]["generic"])
plt.title("Generic - All Groups")
plt.show()

draw_heatmap(corr_specific, config["frames"]["specific"])
plt.title("Specific - All Groups")
plt.show()

draw_heatmap(corr_narrative, config["frames"]["narrative"])
plt.title("Narrative - All Groups")
plt.show()
# %%

# congress correlations
group = "congress"
generic_congress = frame_matrices["generic"][frame_labels["generic"] == group]
specific_congress = frame_matrices["specific"][frame_labels["specific"] == group]
narrative_congress = frame_matrices["narrative"][frame_labels["narrative"] == group]

corr_congress_generic = np.corrcoef(generic_congress.T)
corr_congress_specific = np.corrcoef(specific_congress.T)
corr_congress_narrative = np.corrcoef(narrative_congress.T)

draw_heatmap(corr_congress_generic, config["frames"]["generic"])
plt.title("Generic - Congress")
plt.show()

draw_heatmap(corr_congress_specific, config["frames"]["specific"])
plt.title("Specific - Congress")
plt.show()

draw_heatmap(corr_congress_narrative, config["frames"]["narrative"])
plt.title("Narrative - Congress")
plt.show()
# %%
# congress correlations
group = "journalists"
generic_journalists = frame_matrices["generic"][frame_labels["generic"] == group]
specific_journalists = frame_matrices["specific"][frame_labels["specific"] == group]
narrative_journalists = frame_matrices["narrative"][frame_labels["narrative"] == group]

corr_journalists_generic = np.corrcoef(generic_journalists.T)
corr_journalists_specific = np.corrcoef(specific_journalists.T)
corr_journalists_narrative = np.corrcoef(narrative_journalists.T)

draw_heatmap(corr_journalists_generic, config["frames"]["generic"])
plt.title("Generic - Journalists")
plt.show()

draw_heatmap(corr_journalists_specific, config["frames"]["specific"])
plt.title("Specific - Journalists")
plt.show()

draw_heatmap(corr_journalists_narrative, config["frames"]["narrative"])
plt.title("Narrative - Journalists")
plt.show()
# %%
# congress correlations
group = "trump"
generic_trump = frame_matrices["generic"][frame_labels["generic"] == group]
specific_trump = frame_matrices["specific"][frame_labels["specific"] == group]
narrative_trump = frame_matrices["narrative"][frame_labels["narrative"] == group]

corr_trump_generic = np.corrcoef(generic_trump.T)
corr_trump_specific = np.corrcoef(specific_trump.T)
corr_trump_narrative = np.corrcoef(narrative_trump.T)

draw_heatmap(corr_trump_generic, config["frames"]["generic"])
plt.title("Generic - Trump")
plt.show()

draw_heatmap(corr_trump_specific, config["frames"]["specific"])
plt.title("Specific - Trump")
plt.show()

draw_heatmap(corr_trump_narrative, config["frames"]["narrative"])
plt.title("Narrative - Trump")
plt.show()
# %%
from sklearn.decomposition import PCA

generic_pca = PCA().fit_transform(frame_matrices["generic"])
specific_pca = PCA().fit_transform(frame_matrices["specific"])

# %%
sc = {"congress": {"shape": "o", "color": "C0"},
      "journalists": {"shape": "s", "color": "C1"},
      "trump": {"shape": "^", "color": "C2"},
      "public": {"shape": "x", "color": "C3"}}

fig, ax = plt.subplots(ncols=2, figsize=(10,5))

for group in groups[:-1]:
    gen_group = generic_pca[frame_labels["generic"] == group]
    spe_group = specific_pca[frame_labels["specific"] == group]

    ax[0].scatter(gen_group[:, 0], gen_group[:, 1],
                marker=sc[group]["shape"],
                c=sc[group]["color"],
                label=group, alpha=0.1)

    ax[1].scatter(spe_group[:, 0], spe_group[:, 1],
                  marker=sc[group]["shape"],
                  c=sc[group]["color"],
                  label=group, alpha=0.1)

plt.tight_layout()
plt.show()

# %%
