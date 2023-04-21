import json
import tweet_handler as th
from datetime import datetime

# we're going to stick with 2018 for now
start_date = datetime.strptime("01-01-2018", "%m-%d-%Y")
end_date = datetime.strptime("12-31-2018", "%m-%d-%Y")

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
