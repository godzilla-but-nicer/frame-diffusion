# %%
import gzip
import json
import pandas as pd
import tqdm
from datetime import datetime

# essentially we want to add columns to the annotated data
annotated = pd.read_csv("data/annotated_data/full.tsv", sep="\t", index_col="Unnamed: 0")

# %%

# dumb function to parse out the stuff into a flat dictionary
def parse_twitter_json(json_dict):
    tweet_id = json_dict["id"]
    post_timestamp = datetime.strptime(json_dict["created_at"],
                                       "%a %b %d %H:%M:%S +0000 %Y")
    user_name = json_dict["user"]["screen_name"]


    # responses
    quote_count = json_dict["quote_count"]
    reply_count = json_dict["reply_count"]
    favorite_count = json_dict["favorite_count"]
    retweet_count = json_dict["retweet_count"]
    
    # what is the tweet related to
    if "in_reply_to_status_id" in json_dict:
        reply_to = json_dict["in_reply_to_status_id_str"]
        reply_to_user = json_dict["in_reply_to_screen_name"]
    else:
        reply_to = None
        reply_to_user = None

    if "quoted_status_id" in json_dict:
        quote_of = json_dict["quoted_status_id_str"]
        quote_of_user = json_dict["quoted_status"]["user"]["screen_name"]
    else:
        quote_of = None
        quote_of_user = None
    
    return {"id_str": tweet_id,
            "time_stamp": post_timestamp,
            "screen_name": user_name,
            "quote_count": quote_count,
            "reply_count": reply_count,
            "favorite_count": favorite_count,
            "retweet_count": retweet_count,
            "reply_to_id": reply_to,
            "reply_to_user": reply_to_user,
            "quote_of_id": quote_of,
            "quote_of_user": quote_of_user}


# load the tweets and add them to the list if they match an annotated tweet
tweet_rows = []
id_set = set(annotated["id_str"].astype(str).values)
with gzip.open("data/immigration_tweets_by_country/US.gz", "rb") as zipped:
    for i, file in tqdm.tqdm(enumerate(zipped)):
        tweet_dict = json.loads(file)
        if tweet_dict["id_str"] in id_set:
            tweet_rows.append(parse_twitter_json(tweet_dict))

tweets_df = pd.DataFrame(tweet_rows)
# %%
# ok so now we need to merge onto the annotated data with the retweet info
extended_annotations = pd.merge(annotated, tweets_df, on="id_str")
extended_annotations.to_csv("data/extended_annotated_data/full.tsv",
                            sep="\t", index=False)

# %%
