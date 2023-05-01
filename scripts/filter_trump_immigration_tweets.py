import json
import tweet_handler as th
from datetime import datetime

with open("workflow/config.json", "r") as jf:
    config = json.loads(jf.read())

# we're going to stick with 2018 for now
start_date = datetime.strptime(config["dates"]["start"], "%m-%d-%Y %H:%M:%S")
end_date = datetime.strptime(config["dates"]["end"], "%m-%d-%Y %H:%M:%S")

# same keywords as always
query = "immigration|immigrants?|illegals|undocumented|illegal aliens?|migrants?|migration"

# load all of the trump tweets
with open("data/trump_twitter_archive.json", "r") as fin:
    tweets = json.loads(fin.read())

# pull out the tweets with keywords
keep_tweets = []
for tweet in tweets:
    
    # only check out tweets in our date range
    tweet_date = datetime.strptime(tweet["date"], "%Y-%m-%d %H:%M:%S")

    if tweet_date >= start_date and tweet_date <= end_date:
        filtered_text = th.filter_by_text(tweet, query)

        if filtered_text is not None:
            keep_tweets.append(tweet)

with open("data/immigration_tweets/trump.json", "w") as fout:
    json.dump(keep_tweets, fout)
