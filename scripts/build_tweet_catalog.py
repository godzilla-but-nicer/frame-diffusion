import gzip
import json
import pandas as pd
import numpy as np
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
    tweet_catalog[tweet["id_str"]] = tweet

for tweet_json in tqdm(gzip.open(paths["public"]["2019_json"])):
    tweet = json.loads(tweet_json)
    tweet_catalog[tweet["id_str"]] = tweet

print("Catalogging Journalists tweets")
with open(paths["journalists"]["tweet_json"], "r") as json_file:
    for tweet in tqdm(json.loads(json_file.read())):
        tweet_catalog[str(tweet["id"])] = tweet

print("Catalogging Congress tweets")
with open(paths["congress"]["tweet_json"], "r") as json_file:
    for tweet in tqdm(json.loads(json_file.read())):
        tweet_catalog[str(tweet["id"])] = tweet



with gzip.open("data/immigration_tweets/tweets_by_id.gz", "wb") as tc_fout:
    tc_fout.write(json.dumps(tweet_catalog).encode())
