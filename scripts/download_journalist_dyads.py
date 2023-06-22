# %%
import json
import os
import pandas as pd

import tweet_handler as th

with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

with open("workflow/twitter_api_keys.json", "r") as kf:
    keys = json.loads(kf.read())

# %%
journos = pd.read_csv("data/immigration_tweets/journalists.tsv", sep="\t")

for journalist in journos["screen_name"].unique():

    # subset the tweets for this journalist only
    jtweets = journos[journos["screen_name"] == journalist]

    # lists of all replies and quotes associated with a user
    all_quotes = []
    all_convs = []

    for i, row in jtweets.iterrows():
        
        try:
            conv = th.download_replies(row["id_str"], keys["bearer_token"])
            all_convs.extend(conv)

        except:
            continue
        

        try:
            quotes = th.download_replies(row["id_str"], keys["bearer_token"])
            all_quotes.extend(quotes)

        except:
            continue
        
    with open(f"data/network_tweets/conversations/journalists/{journalist}.json", "r") as fconv:
        fconv.write(json.dumps(all_convs))
    
    with open(f"data/network_tweets/quotes/journalists/{journalist}.json", "r") as fquote:
        fquote.write(json.dumps(all_quotes))
# %%