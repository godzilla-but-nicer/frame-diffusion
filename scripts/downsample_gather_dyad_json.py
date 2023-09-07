import gzip
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import reduce

in_sample_dyads = pd.read_csv("data/down_sample/edge_lists/in_sample_dyads.tsv", sep="\t")

with open("data/down_sample/immigration_tweets/tweets_by_id.json", "r") as tweet_fin:
    tweet_catalog = json.loads(tweet_fin.read())

frame_catalog = pd.read_csv("data/down_sample/binary_frames/all_group_frames.tsv", sep="\t").drop(["text", "Unnamed: 0", "Threat", "Victim", "Hero"], axis="columns")
print("Data Loaded to memory")
frame_catalog["id_str"] = frame_catalog["id_str"].astype(str)
json_list = []
print("Beginning Loop")
for row_i, dyad in in_sample_dyads.iterrows():
    dyad_json = {}
    
    dyad_json["source_full"] = json.loads(tweet_catalog[str(dyad["tweet_id"])])
    dyad_json["target_full"] = json.loads(tweet_catalog[str(dyad["target_id"])])
    # print(f"Attempting to match {type(str(dyad['tweet_id']))} with {type(frame_catalog['id_str'][0])}")
    # print(f"Looks like {str(dyad['tweet_id'])} and {frame_catalog['id_str'][0]}")
    # print(f"Found match: {np.sum((frame_catalog['id_str'] == (dyad['tweet_id']))) > 0}")

    dyad_json["source_frames"] = frame_catalog[frame_catalog["id_str"] == str(dyad["tweet_id"])].to_json(orient="records")
    dyad_json["target_frames"] = frame_catalog[frame_catalog["id_str"] == str(dyad["target_id"])].to_json(orient="records")

    dyad_json["source_group"] = dyad["source_group"]
    dyad_json["target_group"] = dyad["sample"].split("_")[0]  # i stored these weird

    dyad_json["relationship"] = dyad["kind"]

    json_list.append(json.dumps(dyad_json))

print("Writing File")

with open("data/down_sample/edge_lists/full_dyads.json", "w") as fout:
    json.dump(json_list, fout)

print("Complete")