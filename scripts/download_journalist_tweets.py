import json
import tweet_handler as th
import os
import pandas as pd

# get api keys
with open("twitter_api_keys.json", "r") as fin:
    keys = json.loads(fin.read())

# load the journalists
journalists = pd.read_csv("data/users_of_interest/top_536_journos.tsv", sep="\t")

for focal_account in journalists["username"]:

    # already have this one
    if focal_account == "benshapiro":
        continue

    # function handles all api requests
    try:
        user_tweets = th.download_immigration_tweets(focal_account,
                                                     th.start_date,
                                                     th.end_date,
                                                     keys["bearer_token"])
    except:
        continue


    # write out each users tweets
    with open(f"../data/immigration_tweets/journalists/{focal_account}.json", "w") as fout:
        json.dump(user_tweets, fout)
