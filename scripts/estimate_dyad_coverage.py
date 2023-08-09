#%%
import gzip
import json
import pandas as pd

from edge_finder import build_sample_catalog, check_sample_group
from glob import glob
from tqdm import tqdm
from typing import List

import os
if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
print(os.getcwd())


with open("workflow/paths.json", "r") as path_json:
    data_paths = json.loads(path_json.read())

# %%
# let's try it a different way by merging dataframes
# first we need to load some minimal version of the tweet data

def build_minimal_public_df(path):
    
    # build up a list of rows to convert to df
    rows = []

    for tweet in tqdm(gzip.open(path)):
        new_row = {}
        tweet_json = json.loads(tweet)

        new_row["id_str"] = str(tweet_json["id_str"])
        new_row["user_id"] = tweet_json["user"]["id_str"]
        new_row["user_name"] = tweet_json["user"]["screen_name"]

        rows.append(new_row)

    return pd.DataFrame(rows)


def assemble_catalog(dfs: pd.DataFrame, keep_cols=["target_id", "sample"]) -> pd.DataFrame:

    thin_dfs = []
    for df in dfs:
        df_copy = df.copy()
        df_copy["target_id"] = df_copy["id_str"].astype(str)
        thin_dfs.append(df_copy[keep_cols])
    
    catalog = pd.concat(thin_dfs).drop_duplicates()

    return catalog

# ok so we are going to load a bunch of dataframes with the _df suffix as our
# sort of tweet index that we will check our edge lists against

print("loading public data")
public_2018_df = build_minimal_public_df(data_paths["public"]["2018_json"])
public_2018_df["sample"] = "public_2018"
public_2019_df = build_minimal_public_df(data_paths["public"]["2019_json"])
public_2019_df["sample"] = "public_2019"

congress_df = pd.read_csv(data_paths["congress"]["full_tweets"], sep="\t")
congress_df = congress_df[["id_str", "screen_name"]]
congress_df["sample"] = "congress"

journalists_df = pd.read_csv(data_paths["journalists"]["full_tweets"], sep="\t")
journalists_df = journalists_df[["id_str", "screen_name"]]
journalists_df["sample"] = "journalists"

tweet_catalog = assemble_catalog([public_2018_df,
                                  public_2019_df,
                                  congress_df,
                                  journalists_df])

# %%
# we want to save the data by type of edge. these objects will hold the data to save
all_matches = []

# %%
def identify_successors_in_sample(focal_edges: pd.DataFrame,
                                  catalog: pd.DataFrame,
                                  kinds: List[str]) -> pd.DataFrame:
    
    match_dfs = []
    
    for kind in kinds:
        # prepare the edge list for joining
        focal_edges["target_id"] = focal_edges[kind].map(lambda x: "{:.0f}".format(x))
        focal_edges_thin = focal_edges[["tweet_id", "target_id", kind]]

        # filter out rows that dont indicate this kind of edge
        focal_edges_thin = focal_edges_thin[~focal_edges_thin[kind].isna()]

        # we want neater labels for the kind of interaction
        if kind == "retweet_of":
            focal_edges_thin["kind"] = "retweet"
        elif kind == "quote_of":
            focal_edges_thin["kind"] = "quote"
        elif kind == "reply_to":
            focal_edges_thin["kind"] = "reply"
        else:
            raise ValueError(f"Bad kind of interaction: {kind}")

        focal_edges_thin = focal_edges_thin[["tweet_id", "target_id", "kind"]]
        focal_edges_thin["source_group"] = group
        matches = pd.merge(focal_edges_thin, catalog, on="target_id")

        match_dfs.append(matches)

    
    return pd.concat(match_dfs)


# %%
# matches on journalists edges
group = "journalists"
focal_edges = pd.read_csv(data_paths[group]["edges"], sep="\t")
journo_kinds = ["retweet_of", "quote_of", "reply_to"]

match_df = identify_successors_in_sample(focal_edges, tweet_catalog, journo_kinds)

match_sum = match_df.shape[0]
    
print(f"\nJournalists: Found {match_sum} successors (one of {journo_kinds}) in-sample out of {focal_edges.shape[0]} known successors ({match_sum / focal_edges.shape[0] * 100:.2f} %)")

all_matches.append(match_df)
# %%
group = "congress"
focal_edges = pd.read_csv(data_paths[group]["edges"], sep="\t")
congress_kinds = ["retweet_of"]


match_df = identify_successors_in_sample(focal_edges, tweet_catalog, congress_kinds)

match_sum = match_df.shape[0]
    
print(f"\nCongress: Found {match_sum} successors (one of {journo_kinds}) in-sample out of {focal_edges.shape[0]} known successors ({match_sum / focal_edges.shape[0] * 100:.2f} %)")

all_matches.append(match_df)
# %%
group = "public"
edges_2018 = pd.read_csv(data_paths[group]["2018_edges"], sep="\t")
edges_2019 = pd.read_csv(data_paths[group]["2019_edges"], sep="\t")
focal_edges = pd.concat([edges_2018, edges_2019])
public_kinds = ["quote_of", "reply_to"]

match_df = identify_successors_in_sample(focal_edges, tweet_catalog, public_kinds)

match_sum = match_df.shape[0]
    
print(f"\n{group}: Found {match_sum} successors in-sample out of {focal_edges.shape[0]} known successors ({match_sum / focal_edges.shape[0] * 100:.2f} %)")

all_matches.append(match_df)
# %%
all_dyads = pd.concat(all_matches)

all_dyads.to_csv("data/edge_lists/in_sample_dyads.tsv", index=False, sep="\t")
# %%
