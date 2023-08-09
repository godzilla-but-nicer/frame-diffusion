import gzip
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import reduce

in_sample_dyads = pd.read_csv("data/edge_lists/in_sample_dyads.tsv", sep="\t")

with open("workflow/paths.json", "r") as path_file:
    paths = json.loads(path_file.read())


with open("workflow/config.json", "r") as config_file:
    config = json.loads(config_file.read())

tweet_catalog = {}

# this is an annoying part, we have to iterate over each json file
# and put all of the relevant json into the catalog keyed by the id string
print("Catalogging Public tweets")
for tweet_json in tqdm(gzip.open(paths["public"]["2018_json"])):
    tweet = json.loads(tweet_json)
    tweet_catalog[tweet["id_str"]] = json.dumps(tweet)

for tweet_json in tqdm(gzip.open(paths["public"]["2019_json"])):
    tweet = json.loads(tweet_json)
    tweet_catalog[tweet["id_str"]] = json.dumps(tweet)

print("Catalogging Journalists tweets")
with open(paths["journalists"]["tweet_json"], "r") as json_file:
    for tweet in tqdm(json.loads(json_file.read())):
        tweet_catalog[str(tweet["id"])] = json.dumps(tweet)

print("Catalogging Congress tweets")
with open(paths["congress"]["tweet_json"], "r") as json_file:
    for tweet in tqdm(json.loads(json_file.read())):
        tweet_catalog[str(tweet["id"])] = json.dumps(tweet)

with open("data/immigration_tweets/all_tweets_by_id.json", "w") as tc_fout:
    tc_fout.write(json.dumps(tweet_catalog))

frame_catalog = {}

print("Catalogging Public Frames")
pf_df = pd.read_csv("data/binary_frames/predicted_frames.tsv", sep="\t")
for i, tweet in tqdm(pf_df.iterrows()):
    frame_catalog[tweet["id_str"]] = json.dumps(tweet.to_dict())


print("Catalogging Congress frames")
congress_frames_by_type = {}
for frame_type in config["frames"].keys():
        if frame_type != "low_f1":
            congress_frames_by_type[frame_type] = pd.read_csv("data/binary_frames/congress/congress_{frame_type}.tsv", sep="\t")

# for each tweet we get three series---one for each frame type---then we merge
# them into one series of all binary frame labels for each frame type
for i, tweet in tqdm(congress_frames_by_type["generic"].iterrows()):
    frame_series_by_type = []

    for frame_type in config["frames"].keys():

        if frame_type == "low_f1":
            continue
        else:
            cfdf = congress_frames_by_type[frame_type]
            tweet_frames = cfdf[cfdf["id_str"] == tweet["id_str"]]
            frame_series_by_type.append(tweet[config["frames"][frame_type]])
        
    all_frames = reduce(lambda l, r: pd.merge(l, r, on="id_str"), frame_series_by_type)
    frame_catalog[tweet["id_str"]] = json.dumps(all_frames.to_dict())


print("Catalogging Journalist frames")
journalist_frames_by_type = {}
for frame_type in config["frames"].keys():
        if frame_type != "low_f1":
            journalist_frames_by_type[frame_type] = pd.read_csv("data/binary_frames/journalists/journalists_{frame_type}.tsv", sep="\t")

# for each tweet we get three series---one for each frame type---then we merge
# them into one series of all binary frame labels for each frame type
for i, tweet in tqdm(journalist_frames_by_type["generic"].iterrows()):
    frame_series_by_type = []

    for frame_type in config["frames"].keys():

        if frame_type == "low_f1":
            continue
        else:
            jfdf = journalist_frames_by_type[frame_type]
            tweet_frames = jfdf[jfdf["id_str"] == tweet["id_str"]]
            frame_series_by_type.append(tweet[config["frames"][frame_type]])
        
    all_frames = reduce(lambda l, r: pd.merge(l, r, on="id_str"), frame_series_by_type)
    frame_catalog[tweet["id_str"]] = json.dumps(all_frames.to_dict())


# json_list = []
# 
# for row_i, dyad in tqdm(in_sample_dyads.iterrows()):
#     dyad_json = {}
#     
#     dyad_json["source_full"] = tweet_catalog[dyad["tweet_id"]]
#     dyad_json["target_full"] = tweet_catalog[dyad["target_id"]]
# 
#     dyad_json["source_frames"] = frame_catalog[dyad["tweet_id"]]
#     dyad_json["target_frames"] = frame_catalog[dyad["target_id"]]
# 
#     dyad_json["source_group"] = dyad["source_group"]
#     dyad_json["target_group"] = dyad["sample"].split("_")[0]  # i stored these weird
# 
#     dyad_json["relationship"] = dyad["kind"]
# 
#     json_list.append(json.dumps(dyad_json))
# 
# with gzip.open("data/edge_lists/full_dyads.gz", "wb") as fout:
#     fout.write(json_list.encode())
# 