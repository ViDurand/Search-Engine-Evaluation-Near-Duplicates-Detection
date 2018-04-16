import numpy as np
import pandas as pd
import math

#%%
#this function read a 2 column tsv file and convert it into a dictionary using the first column as keys and grouping all volues of second column based on their key
def fromCsvToDict(csv_path, column1, column2):
    df = pd.read_csv(csv_path, sep='\t', header=0)
    keys = list(df[column1])
    values = list(df[column2])
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

#%%
#this function returns the P@k of a query result giving its ground truth and k
def P_at_k(GT, Q_results, k):
     patk = len([i for i in Q_results[0:k] if i in GT])/k
     return patk

#this function returns the mean P@k of a list of query results giving their ground truth and k
def mean_P_at_k(groud_truth_dict, se_results_dict, k):
    patk_list = [] #List of all P@k for all the queries  
    
    for i in groud_truth_dict.keys(): #for each iteration, i will contain the query_id from the ground truth list which is the same as the ones from the query results list
        GT_qi= groud_truth_dict[i] #Ground truth list for the current query
        Qi_results = se_results_dict[i] #Results list for the current query
        
        patk_i = P_at_k(GT_qi, Qi_results, k)
        
        patk_list.append(patk_i)
    
    return np.mean(patk_list)

#%%
#this function returns the R-precision of a query result giving its ground truth and k
def R_precision(GT, Q_results):
     R_precision = len([i for i in Q_results[0:len(GT)] if i in GT])/len(GT)
     return R_precision

#this function returns the list of all R-precision's of a list of query results giving their ground truth and k
def get_r_precision_list(groud_truth_dict, se_results_dict):
    r_precision_list = [] #List of all P@k for all the queries  
    
    for i in groud_truth_dict.keys(): #for each iteration, i will contain the query_id from the ground truth list which is the same as the ones from the query results list
        GT_qi= groud_truth_dict[i] #Ground truth list for the current query
        Qi_results = se_results_dict[i] #Results list for the current query
        
        r_precision_i = R_precision(GT_qi, Qi_results)
        
        r_precision_list.append(r_precision_i)
    
    return r_precision_list

#%%
#this function returns the index of the first relevant result found
def get_index_first_relevant_result(groud_truth_list, se_results_list):
    for index in range(len(se_results_list)):
        if(se_results_list[index] in groud_truth_list):
            return index+1
    return -1 #when there is no relevant result

#this function computes the Mean Reciprocal Rank
def mrr(groud_truth_dict, se_results_dict):
    summation = 0
    for q in se_results_dict.keys():
        index_first_relevant_result = get_index_first_relevant_result(groud_truth_dict[q], se_results_dict[q])
        if (index_first_relevant_result != -1): #this is the case when there is at least one relevant result
            reciprocal_rank_q = 1/index_first_relevant_result
        else: #this is the case when there is no relevant result
            reciprocal_rank_q = 0
        summation = summation + reciprocal_rank_q
    
    N = len(groud_truth_dict.keys())
    return ((1/N) * summation)

#%%
def relevance(doc_id, groud_truth_list):
    if (doc_id in groud_truth_list):
        return 1
    else:
        return 0

def dcg(groud_truth_list, se_results_list, k):
    relevance_1 = relevance(se_results_list[0], groud_truth_list)
    summation = 0
    for position in range(2,k+1):
        summation = summation + (relevance(se_results_list[position-1], groud_truth_list)/math.log2(position))
    
    return relevance_1 + summation
    
#this function
def ndcg(groud_truth_list, se_results_list, k):
    idcg = dcg(groud_truth_list, se_results_list, len(groud_truth_list))
    if (idcg == 0):
        return 0
    return dcg(groud_truth_list, se_results_list, k)/idcg
#this function
def mean_ndcg_at_k(groud_truth_dict, se_results_dict, k):
    ndcg_list = [] #List of all ndcg@k for all the queries  
    
    for i in groud_truth_dict.keys(): #for each iteration, i will contain the query_id from the ground truth list which is the same as the ones from the query results list
        GT_qi= groud_truth_dict[i] #Ground truth list for the current query
        Qi_results = se_results_dict[i] #Results list for the current query
        
        ndcg_i = ndcg(GT_qi, Qi_results, k)
        
        ndcg_list.append(ndcg_i)
    
    return np.mean(ndcg_list)

#%%
#functions for part_1_2

#this function returns the recall of a query result giving it's ground truth
def Recall(GT, Q_results):
     recall = len([i for i in Q_results if i in GT])/len(GT)
     return recall

#this function computes the harmonic mean putting together Precision and Recall. BETA = 1
def F_measure(GT, Q_results, k = 4, beta = 1):
    P = P_at_k(GT, Q_results, k)
    R = Recall(GT, Q_results)
    numerator = (((beta**2) + 1) * P * R)
    denominator = (((beta**2) * P) + R)
    if denominator == 0:
        return 0
    else:
        F = numerator/denominator
    return F

#this function returns the mean P@k of a list of query results giving their ground truth and k    
def mean_F_measure(groud_truth_dict, se_results_dict):
    f_measure_list = [] #List of all f_measures for all the queries  
    
    for i in groud_truth_dict.keys(): #for each iteration, i will contain the query_id from the ground truth list which is the same as the ones from the query results list
        GT_qi = groud_truth_dict[i] #Ground truth list for the current query
        Qi_results = se_results_dict[i][:4] #Results list for the current query
        
        f_measure_i = F_measure(GT_qi, Qi_results)
        f_measure_list.append(f_measure_i)
    
    return np.mean(f_measure_list)