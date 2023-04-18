import pandas as pd

journos = pd.read_csv("data/Top10000Journos.csv")
journos.to_csv("data/top_10k_journos.tsv", sep="\t", index=False)
