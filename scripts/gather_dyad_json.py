import gzip
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import reduce

in_sample_dyads = pd.read_csv("data/edge_lists/in_sample_dyads.tsv", sep="\t")

with open("data/immigration_tweets/tweets_by_id.json", "r") as tweet_fin:
    tweet_catalog = json.loads(tweet_fin.read())

frame_catalog = pd.read_csv("data/binary_frames/all_group_frames.tsv", sep="\t").drop(["text", "Unnamed: 0", "Threat", "Victim", "Hero"], axis="columns")
# frame_catalog["id_str"] = frame_catalog["id_str"]
json_list = []

for row_i, dyad in tqdm(in_sample_dyads.iterrows()):
    dyad_json = {}
    
    dyad_json["source_full"] = json.loads(tweet_catalog[str(dyad["tweet_id"])])
    dyad_json["target_full"] = json.loads(tweet_catalog[str(dyad["target_id"])])

    dyad_json["source_frames"] = frame_catalog[frame_catalog["id_str"] == int(dyad["tweet_id"])].to_dict()
    dyad_json["target_frames"] = frame_catalog[frame_catalog["id_str"] == int(dyad["target_id"])].to_dict()

    dyad_json["source_group"] = dyad["source_group"]
    dyad_json["target_group"] = dyad["sample"].split("_")[0]  # i stored these weird

    dyad_json["relationship"] = dyad["kind"]

    json_list.append(json.dumps(dyad_json))

with open("data/edge_lists/full_dyads.json", "w") as fout:
    json.dump(json_list, fout)
