# %%
import json
import tweet_handler as th
import os


with open("../twitter_api_keys.json", "r") as fin:
    keys = json.loads(fin.read())


focal_account = "benshapiro"
user_tweets = th.download_immigration_tweets(focal_account,
                                th.start_date,
                                th.end_date,
                                keys["bearer_token"])


with open(f"../data/immigration_tweets/journalists/{focal_account}.json", "w") as fout:
    json.dump(user_tweets, fout)
# %%
