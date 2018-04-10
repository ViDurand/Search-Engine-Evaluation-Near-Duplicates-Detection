#%%
import numpy as np
import pandas as pd



#%%
if __name__ == "__main__":
    df = pd.read_csv('../dataset/part_1_1/part_1_1__Ground_Truth.tsv', sep='\t', header=0)
    keys = list(df['Query_id'])
    values = list(df['Relevant_Doc_id'])
    gt = dict() #ground truth dictionary set to empty to each query_id correspond a list of relevant docss
    
    i = 0
    for v in values:
        if keys[i] in gt.keys():
            gt[keys[i]].append(v)
        else:
            gt[keys[i]] = []
            gt[keys[i]].append(v)
        i = i+1
#%%