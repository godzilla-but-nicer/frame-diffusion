import json
import pandas as pd
import sys
import tweet_handler as th
from simpletransformers.classification import MultiLabelClassificationModel as MLCM

# x0 = "RT @EricHolder Administration has constitutional duty to conduct a fair and accurate Census. A citizenship question-not usual-would undermine ability to count immigrant communities/have consequences for redistricting.   Census Bureau must reject this request. Be heard!"
# x1 = "My parents came here on green cards. So did @sundarpichai, @elonmusk, @satyanadella. Trump is saying to immigrants and their kids we don’t have a place in America. It’s not just wrong. It’s dumb. Mr. President, would America really be greater without us?"
# x2 = "here is some text this is the new text QT @realDonaldTrump Democrats are doing nothing for DACA - just interested in politics.  DACA activists and Hispanics will go hard against Dems, will start “falling in love” with Republicans and their President! We are about RESULTS."
# print(th.filter_retweet(x0))
# print(th.filter_retweet(x1))
# print(th.filter_retweet(x2))
# 
# exit(1)

# load the global constants
with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())


# load the correct data according to the command line arg
data_file = f"data/immigration_tweets/{sys.argv[1]}.tsv"
tweets = pd.read_csv(data_file, sep="\t").dropna(subset="text")


# load the models
general = MLCM(model_type=config["model"],
               model_name=config["model_names"]["generic"],
               use_cuda=False)
specific = MLCM(model_type=config["model"],
                model_name=config["model_names"]["specific"],
                use_cuda=False)
narrative = MLCM(model_type=config["model"],
                 model_name=config["model_names"]["narrative"],
                 use_cuda=False)


# just going to use the code from the notebook more or less
models = {}
models["generic"] = general
models["specific"] = specific
models["narrative"] = narrative


# ok we're going to exclude straight retweets so first we'll make a list
ids = []
texts = []
for _, tweet in tweets.iterrows():
    if len(tweet["text"]) > 0:
        pred_text = th.filter_retweet(tweet["text"])
        if pred_text:
            ids.append(tweet["id_str"])
            texts.append(pred_text)

# finally we can do the classification, I'm going to pull the code from notebook
labels = ["generic", "specific", "narrative"]
for frame_type in labels:

    frame_labels = config["frames"][frame_type]
    predictions, raw_outputs = models[frame_type].predict(texts)  #This is where we do multilabel frame classification
    
    # stick all of the predictions into dataframes
    new_df = pd.DataFrame(predictions)
    new_df.columns = frame_labels
    new_df['text'] = texts
    new_df["id_str"] = ids
    new_df = new_df.set_index('id_str')

    # write out each dataframe again using the command line arg
    new_df.to_csv(f"data/binary_frames/{sys.argv[1]}/{sys.argv[1]}_{frame_type}.tsv",
                  sep="\t")
