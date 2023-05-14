# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json as json
from scipy.stats import entropy

with open("../workflow/config.json", "r") as cf:
    config = json.loads(cf.read())


# load data
groups = ["congress", "journalists", "trump", "public"]
frame_types = ["generic", "specific", "narrative"]

frame_sums = {k: {} for k in groups} # will hold a bunch of vectors of frame sums by group

for group in groups:
    for frame_type in frame_types:

        if group != "public":
            predictions = pd.read_csv(f"../data/binary_frames/{group}/{group}_{frame_type}.tsv",
                                      sep="\t")
        else:
            predictions = pd.read_csv(f"../data/binary_frames/predicted_frames.tsv",
                                      sep="\t")
            us_public_ids = pd.read_csv(f"../data/us_public_ids.tsv", sep="\t")
            predictions = pd.merge(us_public_ids, predictions,
                                   on="id_str", how="left")
        
        frame_vec = np.zeros(len(config["frames"][frame_type]))
        for i, frame in enumerate(config["frames"][frame_type]):

            frame_vec[i] = predictions[frame].sum()

        frame_sums[group][frame_type] = frame_vec

# %%
distributions = {}
for frame_type in frame_types:
    type_rows = []
    for group in groups:

        # we just need normalized vectors and group labels for seaborn
        normed_sums = frame_sums[group][frame_type] / np.sum(frame_sums[group][frame_type])
        row_dict = {f: normed_sums[i] for i, f in enumerate(config["frames"][frame_type])}
        row_dict["group"] = group

        type_rows.append(row_dict)
    
    # convert to long for seaborn
    df_wide = pd.DataFrame(type_rows)
    df_long = pd.melt(df_wide, id_vars="group")

    # stick it in the dict for easy access
    distributions[frame_type] = df_long

print(distributions["generic"])

# %%
fig, ax = plt.subplots(ncols=2, figsize=(10, 7))
sns.barplot(distributions["generic"],
            y="variable",
            x="value",
            hue="group",
            ax=ax[0])
        
sns.barplot(distributions["specific"],
            y="variable",
            x="value",
            hue="group",
            ax=ax[1])
ax[0].get_legend().remove()
ax[0].set_ylabel("")
ax[1].set_ylabel("")
ax[0].set_xlabel("Frequency")
ax[1].set_xlabel("Frequency")

plt.tight_layout()
plt.savefig("../plots/frame_eda/frequency_by_group.png")
plt.savefig("../plots/frame_eda/frequency_by_group.pdf")
# %%
fig, ax = plt.subplots(figsize=(4, 6))
sns.barplot(distributions["narrative"],
            y="value",
            x="variable",
            hue="group",
            ax=ax)

ax.set_xlabel("")
ax.set_ylabel("Frequency")
plt.savefig("../plots/frame_eda/narrative_frequency.png")
plt.savefig("../plots/frame_eda/narrative_frequency.pdf")
# %%
diversities = {}
for frame_type in frame_types:
    type_rows = []
    for group in groups:

        # we just need normalized vectors and group labels for seaborn
        normed_sums = frame_sums[group][frame_type] / np.sum(frame_sums[group][frame_type])
        diversity = entropy(normed_sums) / len(config["frames"][frame_type])
        row_dict = {"group": group, "diversity": diversity}
        type_rows.append(row_dict)

    # convert to long for seaborn
    df_wide = pd.DataFrame(type_rows)

    diversities[frame_type] = df_wide

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
sns.barplot(diversities["generic"],
            x="group",
            y="diversity",
            ax=ax[0])
ax[0].set_xlabel("")
ax[0].set_title("Generic Frames")

sns.barplot(diversities["specific"],
            x="group",
            y="diversity",
            ax=ax[1])
ax[1].set_xlabel("")
ax[1].set_title("Specific Frames")

plt.tight_layout()
plt.savefig("../plots/frame_eda/frame_diversity.png")
plt.savefig("../plots/frame_eda/frame_diversity.pdf")
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
