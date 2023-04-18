# %%
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ok load it up baby
df = pd.read_csv("../data/extended_annotated_data/role_labels.tsv",
                 sep="\t")

# %% [markdown]
#
# ## Frame Usage Stats
#
# First let's just look at how often the frames are used. This is not new but
# why not just take a look anyway because why not. Then we'll do new shit.
#
# %%
# ok so first of all I want an accounting of the frames in my sample
general_frames = []
for row in df["Issue-General"].values:
    if type(row) == str and len(row) > 0:
        general_frames.extend(ast.literal_eval(row))

# ok now we can count this stuff
general_vals, general_counts = np.unique(general_frames, return_counts=True)
general_idx = np.argsort(general_counts)

fig, ax = plt.subplots()
ax.set_title("Issue-General Frames Usage")
ax.barh(general_vals[general_idx], general_counts[general_idx])
ax.set_xlabel("Usage Count")
plt.show()

# %%
# ok specific frames now
specific_frames = []
for row in df["Issue-Specific"].values:
    if type(row) == str and len(row) > 0:
        specific_frames.extend(ast.literal_eval(row))

# ok now we can count this stuff
specific_vals, specific_counts = np.unique(specific_frames, return_counts=True)
specific_idx = np.argsort(specific_counts)

fig, ax = plt.subplots()
ax.set_title("Issue-Specific Frames Usage")
ax.barh(specific_vals[specific_idx], specific_counts[specific_idx])
ax.set_xlabel("Usage Count")
plt.show()

# %% [markdown]
#
# ## Ideology Distribution
#
# In our data, how is ideology distributed. If frames are indeed polarized
# then it matters how our sample is identified
#
# %%
fig, ax = plt.subplots()
ax.hist(df["ideology"], bins=50)
plt.show()
# %% [markdown]
# 
# OK so this is really interesting! based on the raw scores we do
# see polarization. specifically, we see a wider spread with a more extreme
# peak on the right and a tighter less extreme distribution on the left.
#
# ## How many journalists in here??

# %%
df[df["role_label"] == "journalist"]
# %%
