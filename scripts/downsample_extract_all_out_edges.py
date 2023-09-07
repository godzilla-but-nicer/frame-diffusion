import pandas as pd
import json
import gzip

from edge_finder import check_connections
from glob import glob
from tqdm import tqdm

with open("workflow/sample_paths.json", "r") as path_file:
    paths = json.loads(path_file.read())

public_2018_path = "data/down_sample/immigration_tweets/US_2018.gz"
public_2019_path = "data/down_sample/immigration_tweets/US_2019.gz"
journalist_paths = "data/down_sample/immigration_tweets/journalists.json"
retweet_paths = glob("data/down_sample/decahose_retweets/decahose*.gz")
congress_path = "data/down_sample/immigration_tweets/congress.json"

print("Extracting Public Edges 2018...")
new_rows = []
for t in tqdm(gzip.open(paths["public"]["2018_json"])):
    tweet_json = json.loads(t)
    edges = check_connections(tweet_json, "public")
    if edges:
        new_rows.append(edges)

pd.DataFrame(new_rows).to_csv("data/down_sample/edge_lists/public_2018_successors.tsv", sep="\t", index=False)

print("Extracting Public Edges 2019...")
new_rows = []
for t in tqdm(gzip.open(paths["public"]["2019_json"])):
    tweet_json = json.loads(t)
    edges = check_connections(tweet_json, "public")
    if edges:
        new_rows.append(edges)

pd.DataFrame(new_rows).to_csv("data/down_sample/edge_lists/public_2019_successors.tsv", sep="\t", index=False)


# print("Extracting Public retweets...")
# new_rows = []
# for daily_path in tqdm(retweet_paths):
#     try:  # at least one of the daily retweet files is corrupted
#         for t in gzip.open(daily_path):
#             tweet_json = json.loads(t)
#             edges = check_connections(tweet_json, "public")
#             if edges:
#                 new_rows.append(edges)
#     except:
#         continue
# 
# pd.DataFrame(new_rows).to_csv("data/down_sample/edge_lists/retweet_successors.tsv", sep="\t", index=False)

print("Extracting Journalist Edges...")
new_rows = []
with open(paths["journalists"]["tweet_json"], "r") as fin:
    for t in tqdm(json.loads(fin.read())):
        edges = check_connections(t, "journalists")
        if edges:
            new_rows.append(edges)

pd.DataFrame(new_rows).to_csv("data/down_sample/edge_lists/journalists_successors.tsv", sep="\t", index=False)

print("Extracting Congress Edges")
with open(paths["congress"]["tweet_json"], "r") as fin:
    for t in tqdm(json.loads(fin.read())):
        edges = check_connections(t, "congress")
        if edges:
            new_rows.append(edges)

pd.DataFrame(new_rows).to_csv("data/down_sample/edge_lists/congress_successors.tsv", sep="\t", index=False)
