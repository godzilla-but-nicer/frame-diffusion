import gzip
import json
import time
import tweet_handler as th
import glob


# get api keys
with open("workflow/twitter_api_keys.json", "r") as fin:
    keys = json.loads(fin.read())

# get the list of conversations


def parse_id(x): return x.split("/")[-1].split(".")[0]


downloaded_convs = [parse_id(file) for file in glob.glob(
    "data/immigration_tweets/conversations/*")]

for conversation in downloaded_convs[:10]:

    with gzip.open(f"data/immigration_tweets/conversations/{conversation}.gz", "r") as fin:
        content = fin.read()
        if len(content) > 3:
            try:
                quotes = th.download_quote_tweets(
                    conversation, keys["bearer_token"])
                print(json.dumps(quotes))
            except:
                continue
