#%%
import numpy as np
import pandas as pd

def fromCsvToDict_forGT(csv_path):
    df = pd.read_csv(csv_path, sep='\t', header=0)
    keys = list(df['Query_id'])
    values = list(df['Relevant_Doc_id'])
    dic = dict() #ground truth dictionary set to empty to each query_id correspond a list of relevant docss
    
    i = 0
    for v in values:
        if keys[i] in dic.keys():
            dic[keys[i]].append(v)
        else:
            dic[keys[i]] = []
            dic[keys[i]].append(v)
        i = i+1
        
    return dic

def fromCsvToDict_forSE(csv_path):
    df = pd.read_csv(csv_path, sep='\t', header=0)
    keys = list(df['Query_ID'])
    doc_ids = list(df['Doc_ID'])
    ranks = list(df['Rank'])
    dic = dict() #ground truth dictionary set to empty to each query_id correspond a list of relevant docss
    
    for i in range(len(keys)):
        if keys[i] in dic.keys():
            dic[keys[i]].append((doc_ids[i], ranks[i]))
        else:
            dic[keys[i]] = []
            dic[keys[i]].append((doc_ids[i], ranks[i]))
        i = i+1
        
    return dic
#%%
if __name__ == "__main__":
    ground_truth_path = '../dataset/part_1_1/part_1_1__Ground_Truth.tsv'
    Results_SE_1_path = '../dataset/part_1_1/part_1_1__Results_SE_1.tsv'
    Results_SE_2_path = '../dataset/part_1_1/part_1_1__Results_SE_2.tsv'
    Results_SE_3_path = '../dataset/part_1_1/part_1_1__Results_SE_3.tsv'
    
    gt_part1 = fromCsvToDict_forGT(ground_truth_path)
    se1_part1 = fromCsvToDict_forSE(Results_SE_1_path)
    
    
#%%s