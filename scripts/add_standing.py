# %%
import pandas as pd

tweet_data = pd.read_csv("../data/extended_annotated_data/full.tsv",
                         sep="\t",
                         index_col="Unnamed: 0.1")
journalists = pd.read_csv("../data/top_10k_journos.tsv",
                          sep="\t")

# %%
journalists["role_label"] = "journalist"
journalists["screen_name"] = journalists["username"]
journalist_labels = journalists[["screen_name", "role_label"]]

labelled_data = pd.merge(tweet_data, journalist_labels,
                         how="left",
                         on="screen_name")
labelled_data["role_label"] = labelled_data["role_label"].fillna("public")
labelled_data.to_csv("../data/extended_annotated_data/role_labels.tsv",
                     sep="\t",
                     index=False)
# %%
