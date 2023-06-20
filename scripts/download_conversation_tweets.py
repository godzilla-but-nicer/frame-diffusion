import gzip
import json
import time
import tweet_handler as th
import glob


# get api keys
with open("workflow/twitter_api_keys.json", "r") as fin:
    keys = json.loads(fin.read())

years = ["2018", "2019"]

# figure out what we have already
parse_id = lambda x: x.split("/")[-1].split(".")[0]
downloaded_convs = [parse_id(file) for file in  glob.glob("data/immigration_tweets/conversations/*")]
checked = set(downloaded_convs)

for year in years:
    with gzip.open(f"data/immigration_tweets/US_{year}.gz", "r") as fgz:
        for i, line in enumerate(fgz):

            get_id = False
            obj = json.loads(line.decode('utf-8').strip())

            if obj["in_reply_to_status_id"] != "null":
                get_id = obj["in_reply_to_status_id"]

            if get_id and get_id not in checked:
                try:
                    conversation_tweets = th.download_conversation(get_id, keys["bearer_token"], immigration_only=False)
                    checked.add(get_id)

                    with gzip.open(f"data/immigration_tweets/conversations/{get_id}.gz", "w") as fout:
                        fout.write(json.dumps(conversation_tweets).encode())
                        fout.flush()
                except:
                    continue

