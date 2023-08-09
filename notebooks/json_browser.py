# %%
import json
import gzip

data_path = "../data/immigration_tweets/US_2018.gz"
print("Example Public JSON")
for t in gzip.open(data_path):
    json_stuff = json.loads(t)
    if json_stuff["is_quote_status"]:
        print(json.dumps(json_stuff, indent=4))
        break
# %%
journalist_data = "../data/immigration_tweets/journalists/jdawsey1.json"

counter = 0
with open(journalist_data, "r") as fin:
    for t in json.loads(fin.read()):
        if "referenced_tweets" in t:
            if t["referenced_tweets"][0]["type"] != "quoted":
                print(json.dumps(t, indent=4))
                counter += 1
                if counter > 10:
                    break
# %%
congress_tweets = "../data/immigration_tweets/congress.json"
counter = 0
with open(congress_tweets, "r") as fin:
    for t in json.loads(fin.read()):
        print(json.dumps(t, indent=2))
        counter += 1
        if counter > 20:
            break
# %%
