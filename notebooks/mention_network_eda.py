# %%
import numpy as np
import networkx as nx
import pandas as pd

import os
if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
print(f"Working in {os.getcwd()}")

raw_edges = pd.read_csv("data/down_sample/edge_lists/in_sample_mentions.tsv", sep="\t")

#%%
back_edges = pd.DataFrame()
back_edges["source"] = raw_edges["uid2"]
back_edges["target"] = raw_edges["uid1"]
back_edges["weight"] = raw_edges["2to1freq"]

forw_edges = pd.DataFrame()
forw_edges["source"] = raw_edges["uid1"]
forw_edges["target"] = raw_edges["uid2"]
forw_edges["weight"] = raw_edges["1to2freq"]

edges = pd.concat([forw_edges, back_edges])
sampled_edges = edges.sample(frac=0.1)

sampled_edges.to_csv("data/down_sample/edge_lists/in_sampe_mentions_longer.csv", index=False)

mention = nx.from_pandas_edgelist(sampled_edges, edge_attr="weight", create_using=nx.DiGraph)
nx.write_graphml(mention, "data/down_sample/edge_lists/in_sample_mentions.graphml")

# %%
import graph_tool.all as gt
import pickle

# %%
if True:
    with open("data/down_sample/edge_lists/gt_community_structure.pkl", "wb") as fout:
        g = gt.load_graph_from_csv("data/down_sample/edge_lists/in_sampe_mentions_longer.csv",
                               skip_first=True,
                               eprop_types=["int"],
                               directed=True)


        # calculate the community structure
        sbm = gt.minimize_nested_blockmodel_dl(g)


        pickle.dump(sbm, fout)

# %%

import matplotlib.pyplot as plt
# calculate a layout for the graph
sfdp_pos = gt.sfdp_layout(g, eweight=g.edge_properties["weight"])

# with open("data/down_sample/edge_lists/gt_community_structure.pkl", "rb") as fin:
#     sbm = pickle.load(fin)

# %%

levels = sbm.get_levels()
for level in levels:
    print(level)

# %%
sbm.draw()
# %%
twelve_blocks = levels[2]
five_blocks = levels[3]
two_blocks = levels[4]
# %%
tb_map = twelve_blocks.get_blocks()
# %%
pr_map = gt.pagerank(g, weight=g.edge_properties["weight"])
# %%
# lets try to identify the highest pagerank in each block
pageranks = pr_map.get_array()
block_labels = tb_map.get_array()
central_indices = []
for block in np.unique(block_labels):
    max_pr_in_block = np.max(pageranks[block_labels == block])
    central_indices.append(np.argwhere((pageranks == max_pr_in_block) & (block_labels == block)))

# %%

# %%
