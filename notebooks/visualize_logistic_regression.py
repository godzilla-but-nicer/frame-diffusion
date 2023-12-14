# %%
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

import os
if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
print(f"Working in {os.getcwd()}")

# config has frame names
print("loading config and paths")
with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

# paths to data files ( need to fix this)
with open("workflow/paths.json", "r") as pf:
    paths = json.loads(pf.read())
print("config and paths loaded")

# load the results dictionary
with open(paths["regression"]["result_pickles"] + "self_influence.pkl", "rb") as self_pkl:
    self_results = pickle.load(self_pkl)
# %%
frame = "Economic"

