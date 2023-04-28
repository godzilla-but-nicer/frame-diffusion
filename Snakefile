import pandas as pd
import json

configfile: "workflow/config.yaml"

journalists = pd.read_csv("data/users_of_interest/top_356_journos.tsv", sep="\t")["username"]

rule download_congress_tweets:
    input:
        "scripts/download_congress_immigration_tweets.py"
    output:
        "data/immigration_tweets/congress.json"
    script:
        "scripts/download_congress_immigration_tweets.py"


rule download_journalist_tweets:
    input:
        "top_356_journos.tsv"
    output:
        expand("data/immigration_tweets/journalists/{journalist}.json", journalist=journalists)
    script:
        "scripts/download_journalist_tweets.py"


rule filter_trump_immigration:
    input:
        "data/trump_twitter_archive.json"
    output:
        "data/immigration_tweets/trump.json"
    script:
        "scripts/filter_trump_immigration.py"


rule build_congress_table:
    input:
        "data/immigration_tweets/congress.json"
    output:
        "data/immigration_tweets/congress.tsv"
    run:
        with open(input, "r") as json_file:
            tweets = json.loads(json_file.read())
        
        df = pd.DataFrame(tweets)
        with open(output, "w") as tsv_file:
            df.to_csv(tsv_file, sep="\t", index=False)


rule build_trump_table:
    input:
        "data/immigration_tweets/trump.json"
    output:
        "data/immigration_tweets/trump.tsv"
    run:
        with open(input, "r") as json_file:
            tweets = json.loads(json_file.read())
        
        df = pd.DataFrame(tweets)
        with open(output, "w") as tsv_file:
            df.to_csv(tsv_file, sep="\t", index=False)

rule build_journalist_table:
    input:
        expand("data/immigration_tweets/{journalist}.json", journalist=journalists)
    output:
        "data/immigration_tweets/journalists.tsv"
    run:
        with open(input, "r") as json_file:
            tweets = json.loads(json_file.read())
        
        df = pd.DataFrame(tweets)
        with open(output, "w") as tsv_file:
            df.to_csv(tsv_file, sep="\t", index=False)


