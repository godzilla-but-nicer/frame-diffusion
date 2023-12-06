# %%
import numpy as np
import pandas as pd
import json

from tqdm import tqdm
from typing import Dict

import os
if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
print(f"Working in {os.getcwd()}")

with open("workflow/paths.json", "r") as pf:
    paths = json.loads(pf.read())

def extract_user(tweet: Dict) -> str:
    if "user_id" in tweet:
        return tweet['user_id']
    elif "user" in tweet:
        return tweet['user']['id']
    elif "author_id":
        return tweet['author_id']
    else:
        raise KeyError(f"Standard keys not present in tweet: {tweet}")

# %%
# get all of the uids for users in our sample
with open(paths["tweet_catalog"], "r") as all_tweets:
    tweets = json.loads(all_tweets.read())


all_users = set([])

for tweet_id in tqdm(tweets.keys()):
    tweet = json.loads(tweets[tweet_id])
    all_users.add(extract_user(tweet))

# %%
mentions = pd.read_csv(paths["mentions"]["raw_network"], sep="\t")

in_sample_mentions = mentions[(mentions["uid1"].isin(all_users)) & (mentions["uid2"].isin(all_users))]
in_sample_mentions.to_csv(paths["mentions"]["network"], index=False, sep="\t")
# %%
