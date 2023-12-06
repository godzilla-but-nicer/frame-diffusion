import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

import os
if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
print(f"Working in {os.getcwd()}")

with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

with open("workflow/paths.json", "r") as pf:
    paths = json.loads(pf.read())
import gzip

user_map = []

years = ["2018", "2019"]
for year in years:
    for tweet_str in gzip.open(paths["public"][year + "_json"]):
        tweet = json.loads(tweet_str)

        if paths["dataset"] == "full":
                user = {}
                user["screen_name"] = tweet["user"]["screen_name"]
                user["user_id"] = tweet["user"]["id"]
                user_map.append(user)
        else:
            for t in tweet:
                user = {}
                user["screen_name"] = tweet["user"]["screen_name"]
                user["user_id"] = tweet["user"]["id"]
                user_map.append(user)

df = pd.DataFrame(user_map).drop_duplicates()
df.to_csv(paths["public"]["user_id_map"], sep="\t", index=False)
