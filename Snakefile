import gzip
import json
import numpy as np
import pandas as pd
import tweet_handler as th

from glob import glob
from tqdm import tqdm

configfile: "workflow/config.json"

journalists = pd.read_csv("data/user_info/top_536_journos.tsv", sep="\t")["username"]

# 1. Download/extract the immigration tweet json data
# no api request
rule download_congress_tweets:
    input:
        "scripts/download_congress_immigration_tweets.py"
    output:
        "data/immigration_tweets/congress.json"
    script:
        "scripts/download_congress_immigration_tweets.py"


# takes multiple days, thousands of api requests
# elon destroyed this step of the workflow
rule download_journalist_tweets:
    input:
        "data/user_info/top_536_journos.tsv"
    output:
        expand("data/immigration_tweets/journalists/{journalist}.json", journalist=journalists)
    script:
        "scripts/download_journalist_tweets.py"


# pulling out the immigration subset from trump twitter archive
rule filter_trump_immigration:
    input:
        "data/trump_twitter_archive.json"
    output:
        "data/immigration_tweets/trump.json"
    script:
        "scripts/filter_trump_immigration_tweets.py"



rule get_us_tweet_ids_times:
    input:
        "data/immigration_tweets/US_2018.gz",
        "data/immigration_tweets/US_2019.gz"
    output:
        "data/us_public_ids.tsv"
    run:
        ids = []
        time_stamps = []
        screen_names = []
        for i in range(2):
            with gzip.open(input[i], 'r') as f:
                for i, line in enumerate(f):
                    tweet = th.load_tweet_obj(line)
                    ids.append(tweet["id_str"])
                    time_stamps.append(tweet["created_at"])
                    screen_names.append(tweet["user"]["screen_name"])

        
        df = pd.DataFrame({"id_str": ids, "time_stamp": time_stamps, "screen_name": screen_names})
        df.to_csv(output[0], sep="\t", index=False)


# 2. convert the json to tsv for each main category we're introducing
rule build_public_sample_table:
    input:
        "data/immigration_tweets/US.gz",
        "data/us_public_ids.tsv"
    output:
        "data/immigration_tweets/public_sample.tsv"
    script:
        "scripts/take_public_sample.py"


rule build_congress_table:
    input:
        "data/immigration_tweets/congress.json"
    output:
        "data/immigration_tweets/congress.tsv"
    run:
        with open(input[0], "r") as json_file:
            tweets = json.loads(json_file.read())

        df_rows = []
        for tweet in tweets:
            row = th.parse_tweet_json(tweet, "congress")
            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)
        with open(output[0], "w") as tsv_file:
            df.to_csv(tsv_file, sep="\t", index=False)


rule build_congress_nonimmigration_table:
    input:
        "data/non_immigration_tweets/congress.json"
    output:
        "data/non_immigration_tweets/congress.tsv"
    run:
        with open(input[0], "r") as json_file:
            tweets = json.loads(json_file.read())

        df_rows = []
        for tweet in tweets:
            if "id" not in tweet:
                continue
            row = th.parse_tweet_json(tweet, "congress")
            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)
        with open(output[0], "w") as tsv_file:
            df.to_csv(tsv_file, sep="\t", index=False)


rule build_trump_table:
    input:
        "data/immigration_tweets/trump.json"
    output:
        "data/immigration_tweets/trump.tsv"
    run:
        with open(input[0], "r") as json_file:
            tweets = json.loads(json_file.read())
        
        df_rows = []
        for tweet in tweets:
            row = th.parse_tweet_json(tweet, "trump")
            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)
        with open(output[0], "w") as tsv_file:
            df.to_csv(tsv_file, sep="\t", index=False)


rule build_journalist_table:
    input:
        expand("data/immigration_tweets/journalists/{journalist}.json", journalist=journalists)
    output:
        "data/immigration_tweets/journalists.tsv"
    run:
        tweets = []
        for inp in input:
            with open(inp, "r") as json_file:
                tweets.extend(json.loads(json_file.read()))

        df_rows = []
        for tweet in tweets:
            row = th.parse_tweet_json(tweet, "v2")
            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)
        with open(output[0], "w") as tsv_file:
            df.to_csv(tsv_file, sep="\t", index=False)

rule build_retweet_table:
    input:
        "data/decahose_retweets/"
    output:
        "data/decahose_retweets/{year}_retweets.tsv"
    run:
        input_files = glob(input[0] + f"/decahose.{wildcards.year}*.gz")

        all_df_rows = []
        for f in input_files:
            
            try:
                new_df_rows = []
                for json_str in gzip.open(f):
                    tweet = json.loads(json_str)
                    row = th.parse_tweet_json(tweet, "v1_retweet")
                    new_df_rows.append(row)
                
                all_df_rows.extend(new_df_rows)
            
            except:
                continue
            
        
        df = pd.DataFrame(all_df_rows)
        with open(output[0], "w") as tsv_file:
            df.to_csv(tsv_file, sep="\t", index=False)


rule collect_retweet_json:
    input:
        "data/decahose_retweets/"
    output:
        "data/decahose_retweets/{year}_retweets.gz"
    run:
        gzip_files = glob(f"data/decahose_retweets/decahose.{wildcards.year}-*.gz")

        json_entries = []
        for file in tqdm(gzip_files):
            try:
                for tweet in gzip.open(file):
                    part_json = json.loads(tweet)
                    json_entries.extend(part_json)

            except:
                print(f"File may be corrupted: {file}")
                continue


        with gzip.open(f"data/decahose_retweets/{wildcards.year}_retweets.gz", "wt", encoding="UTF-8") as gz_fout:
            json.dump(json_entries, gz_fout)


# 3. Use classifier to identify frames in our tweet data
rule classify_congress_tweets:
    input:
        "data/immigration_tweets/congress.tsv"
    output:
        "data/binary_frames/congress/congress_generic.tsv",
        "data/binary_frames/congress/congress_specific.tsv",
        "data/binary_frames/congress/congress_narrative.tsv"
    shell:
        "python scripts/classify_tweet_frames.py congress"


rule classify_trump_tweets:
    input:
        "data/immigration_tweets/trump.tsv"
    output:
        "data/binary_frames/trump/trump_generic.tsv",
        "data/binary_frames/trump/trump_specific.tsv",
        "data/binary_frames/trump/trump_narrative.tsv"
    shell:
        "python scripts/classify_tweet_frames.py trump"


rule classify_journalist_tweets:
    input:
        "data/immigration_tweets/journalists.tsv"
    output:
        "data/binary_frames/journalists/journalists_generic.tsv",
        "data/binary_frames/journalists/journalists_specific.tsv",
        "data/binary_frames/journalists/journalists_narrative.tsv"
    shell:
        "python scripts/classify_tweet_frames.py journalists"


rule classify_retweets:
    input:
        expand("data/decahose_retweets/{year}_retweets.tsv", year=["2018", "2019"])
    output:
        "data/binary_frames/retweets/retweets_generic.tsv",
        "data/binary_frames/retweets/retweets_specific.tsv",
        "data/binary_frames/retweets/retweets_narrative.tsv"
    shell:
        "python scripts/classify_tweet_frames.py retweets"


rule identify_successors:
    input:
        "data/immigration_tweets/US_2018.gz",
        "data/immigration_tweets/US_2019.gz",
        "data/immigration_tweets/journalists.json",
        "data/immigration_tweets/congress.json",
        "data/decahose_retweets/"
    output:
        "data/edge_lists/public_2018_successors.tsv",
        "data/edge_lists/public_2019_successors.tsv",
        "data/edge_lists/journalists_successors.tsv",
        "data/edge_lists/congress_successors.tsv",
        "data/edge_lists/retweet_successors.tsv"

    shell:
        "python scripts/extract_all_out_edges.py"

