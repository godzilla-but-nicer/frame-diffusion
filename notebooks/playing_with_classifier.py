# %%
import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel

# %%
# If running on server with GPUs, can set use_cuda=True
model_generic = MultiLabelClassificationModel(model_type='roberta',model_name="juliamendelsohn/framing_issue_generic",use_cuda=False)
model_specific = MultiLabelClassificationModel(model_type='roberta',model_name="juliamendelsohn/framing_immigration_specific",use_cuda=False)
model_narrative = MultiLabelClassificationModel(model_type='roberta',model_name="juliamendelsohn/framing_narrative",use_cuda=False)

# %%
# All frame labels are in alphabetical order by category. We put these together with predictions to understand results
frames = {}
frames['Issue-Generic'] = ['Capacity and Resources', 'Crime and Punishment','Cultural Identity', 
'Economic','External Regulation and Reputation', 'Fairness and Equality','Health and Safety',
'Legality, Constitutionality, Jurisdiction', 'Morality and Ethics','Policy Prescription and Evaluation',
'Political Factors and Implications', 'Public Sentiment','Quality of Life', 'Security and Defense']
frames['Issue-Specific'] = ['Hero: Cultural Diversity','Hero: Integration', 'Hero: Worker',
'Threat: Fiscal', 'Threat: Jobs', 'Threat: National Cohesion','Threat: Public Order',
'Victim: Discrimination','Victim: Global Economy', 'Victim: Humanitarian', 'Victim: War']
frames['Narrative'] = ['Episodic','Thematic']

# %%
models = {}
models['Issue-Generic'] = model_generic
models['Issue-Specific'] = model_specific
models['Narrative'] = model_narrative

# %%
#Example messages about immigration (not from our dataset)
texts = []
texts.append("We are a nation of immigrants. Our diversity is our greatest strength.")
texts.append("Illegal immigrants steal our money.")
texts.append("Twenty migrants died when their boat sank in the Mediterranean.")

# %%
#Loops through all 3 frame types (Issue-Generic, Issue-Specific, Narrative)
#Produces list of Pandas dataframes (for each frame type) with binary indicator for if a frame is detected
dfs = []
for frame_type in frames.keys():
    frame_labels = frames[frame_type]
    predictions, raw_outputs = models[frame_type].predict(texts)  #This is where we do multilabel frame classification
    new_df = pd.DataFrame(predictions)
    new_df.columns = frame_labels
    new_df['Text'] = texts
    new_df = new_df.set_index('Text')
    dfs.append(new_df)
# %%
dfs[1]
# %%
