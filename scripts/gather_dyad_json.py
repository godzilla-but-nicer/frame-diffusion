import gzip
import json
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from typing import Dict
from functools import reduce

# convert ugly named series to nice little dict
def frame_series_to_dict(frame_series: pd.Series, prefix: str) -> Dict:
    old_dict = frame_series.to_dict()
    new_dict = {}
    for key in old_dict.keys():
        new_key = re.sub(rf"^{prefix}", "", key)
        new_dict[new_key] = old_dict[key]
    
    return new_dict


in_sample_dyads = pd.read_csv("data/edge_lists/in_sample_dyads.tsv", sep="\t")

with open("data/immigration_tweets/tweets_by_id.json", "r") as tweet_fin:
    tweet_catalog = json.loads(tweet_fin.read())

frame_catalog = pd.read_csv("data/binary_frames/all_group_frames.tsv", sep="\t").drop(["text", "Unnamed: 0", "Threat", "Victim", "Hero"], axis="columns")
print("Data Loaded to memory")


# make a version of frame df with id str renamed tweet id for merging
# also one with target_id
print("Collecting frames")
source_frames = (frame_catalog
                 .add_prefix("s_")
                 .rename({"s_id_str": "tweet_id"}, axis="columns"))
target_frames = (frame_catalog
                 .add_prefix("t_")
                 .rename({"t_id_str": "target_id"}, axis="columns"))

dyads_source = pd.merge(in_sample_dyads, source_frames, on="tweet_id")
dyads_target = pd.merge(in_sample_dyads, target_frames, on="target_id")
dyads_framed = pd.merge(dyads_source, dyads_target, on=["tweet_id", "target_id", "kind", "source_group", "sample"])
dyads_framed = pd.concat([dyads_framed, dyads_target[dyads_target["kind"] == "retweet"]])

# lists of frame columns
source_frame_cols = [col for col in dyads_framed if "s_" in col]
target_frame_cols = [col for col in dyads_framed if "t_" in col]

json_list = []
print("Beginning Loop")
for row_i, dyad in dyads_framed.iterrows():
    dyad_json = {}
    
    dyad_json["source_full"] = json.loads(tweet_catalog[str(dyad["tweet_id"])])
    dyad_json["target_full"] = json.loads(tweet_catalog[str(dyad["target_id"])])
    
    source_frames = dyad[source_frame_cols]
    source_frames = frame_series_to_dict(source_frames, "s_")
    target_frames = dyad[target_frame_cols]
    target_frames = frame_series_to_dict(target_frames, "t_")

    dyad_json["source_frames"] = source_frames
    dyad_json["target_frames"] = target_frames

    dyad_json["source_group"] = dyad["source_group"]
    dyad_json["target_group"] = dyad["sample"].split("_")[0]  # i stored these weird

    dyad_json["relationship"] = dyad["kind"]

    json_list.append(json.dumps(dyad_json))

print("Writing File")

with open("data/edge_lists/full_dyads.json", "w") as fout:
    json.dump(json_list, fout)

print("Complete")