import gzip
import json
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from functools import reduce


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


print("Catalogging Retweets")
for file in tqdm(glob(paths["public"]["retweet_dir"] + "decahose.*.gz")):
    for tweet_json in gzip.open(file):
        for tweet in json.loads(tweet_json):
            tweet_catalog[str(tweet["id_str"])] = json.dumps(tweet)


print("Writing file")
with open("data/immigration_tweets/tweets_by_id.json", "w") as tc_fout:
    tc_fout.write("{\n")
    for id_str in tqdm(tweet_catalog.keys()):
        tc_fout.write(f'"{id_str}": {tweet_catalog[id_str]},\n')
    tc_fout.write("}\n")
