import gzip
import json
import time
import tweet_handler as th

from tqdm import tqdm

# get api keys
with open("workflow/twitter_api_keys.json", "r") as fin:
    keys = json.loads(fin.read())

years = ["2018", "2019"]
checked = set({})

for year in years:
    yearly_conversations = []
    with gzip.open(f"data/immigration_tweets/US_{year}.gz", "r") as fgz:
        for i, line in enumerate(fgz):

            get_id = False
            obj = json.loads(line.decode('utf-8').strip())

            if obj["in_reply_to_status_id"] != "null":
                get_id = obj["in_reply_to_status_id"]

            if get_id and get_id not in checked:
                try:
                    conversation_tweets = th.download_conversation(get_id, keys["bearer_token"])
                    checked.add(get_id)
                    yearly_conversations.extend(conversation_tweets)
                    time.sleep(1)
                except:
                    continue

    with gzip.open(f"data/immigration_tweets/US_conversations_{year}.gz", "w") as fout:
        fout.write(json.dumps(yearly_conversations).encode())
