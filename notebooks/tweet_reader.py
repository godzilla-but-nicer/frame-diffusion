# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json as json
import textwrap
import datetime
from scipy.stats import entropy
import os

if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")

import data_selector as ds

with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

np.random.seed(2133)
# %%
journalists = ds.load_journalist_frames()
journalists["time_stamp"] = pd.to_datetime(journalists["time_stamp"], format="%Y-%m-%d %H:%M:%S", utc=True)

def sample_tweets_by_date(df, date):
    dated = df[df["time_stamp"].dt.date == date]

    try:
        sample = dated.sample(10)
    except:
        sample = dated

    print(f"{date}")
    print("-"*70 + "\n")
    for sample_text in sample["text"]:
        lines = textwrap.wrap(f"{sample_text}", 
                                70, 
                                subsequent_indent="        ", 
                                initial_indent="--->  ")

        for line in lines:
            print(line)

# %%
focal_date = datetime.date(2019, 9, 13)
sample_tweets_by_date(journalists, focal_date)
# %%
