# %%
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.linalg as sla
import tqdm
from simpletransformers.classification import MultiLabelClassificationModel as MLCM


with open("../data/edge_lists/full_dyads.json") as json_file:
    dyads = json.loads(json_file.read())

with open("../workflow/config.json") as config_file:
    config = json.loads(config_file.read())

print(len(dyads))

# %%
general = MLCM(model_type=config["model"],
               model_name=config["model_names"]["generic"],
               use_cuda=False)
specific = MLCM(model_type=config["model"],
                model_name=config["model_names"]["specific"],
                use_cuda=False)
narrative = MLCM(model_type=config["model"],
                 model_name=config["model_names"]["narrative"],
                 use_cuda=False)

models = {}
models["generic"] = general
models["specific"] = specific
models["narrative"] = narrative

# %%
frame_dyad = {}
for i, dyad in enumerate(tqdm.tqdm(dyads)):
    obj = json.loads(dyad)
    if obj["relationship"] == "retweet":

        # classify the target only
        target_generic_frames = general.predict([obj["target_full"]["text"]])[0]
        target_specific_frames = specific.predict([obj["target_full"]["text"]])[0]
        target_narrative_frames = specific.predict([obj["target_full"]["text"]])[0]

    if i > 5:
        break




print(hits / denominator)    
# %%
frame_catalog["id_str"]

# %%
