import gzip
import json
import pandas as pd
import numpy as np
import sys
from glob import glob
from tqdm import tqdm
from functools import reduce


with open("workflow/sample_paths.json", "r") as path_file:
    paths = json.loads(path_file.read())

with open("workflow/config.json", "r") as config_file:
    config = json.loads(config_file.read())

tweet_catalog = {}

# this is an annoying part, we have to iterate over each json file
# and put all of the relevant json into the catalog keyed by the id string
if len(sys.argv) == 1: 
    print("Catalogging Public tweets")
    for tweet_json in tqdm(gzip.open(paths["public"]["2018_json"])):
        tweets = json.loads(tweet_json)
        for tweet in tweets:
            tweet_catalog[tweet["id_str"]] = json.dumps(tweet)

    for tweet_json in tqdm(gzip.open(paths["public"]["2019_json"])):
        tweets = json.loads(tweet_json)
        for tweet in tweets:
            tweet_catalog[tweet["id_str"]] = json.dumps(tweet)


    print("Catalogging Journalists tweets")
    with open(paths["journalists"]["tweet_json"], "r") as json_file:
        for tweet in tqdm(json.loads(json_file.read())):
            tweet_catalog[str(tweet["id"])] = json.dumps(tweet)


    print("Catalogging Congress tweets")
    with open(paths["congress"]["tweet_json"], "r") as json_file:
        for tweet in tqdm(json.loads(json_file.read())):
            tweet_catalog[str(tweet["id"])] = json.dumps(tweet)

    print("Writing file")
    with open("data/down_sample/immigration_tweets/tweets_by_id.json", "w") as tc_fout:
        json.dump(tweet_catalog, tc_fout)

elif sys.argv[1] == "retweets":
    print("Catalogging Retweets")
    year = sys.argv[3]
    files = glob(paths["public"]["retweet_dir"] + f"decahose.{year}*.gz")
    for file in tqdm(files):
        try:
            for tweet_json in gzip.open(file):
                tweet = json.loads(tweet_json)
                tweet_catalog[str(tweet["id_str"])] = json.dumps(tweet)
        except:
            continue

    print("Writing file")
    with open(sys.argv[2], "w") as tc_fout:
        json.dump(tweet_catalog, tc_fout)
