#%%
import numpy as np
import pandas as pd
import math

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

#this function returns the P@k of a query result giving its ground truth and k
def P_at_k(GT, Q_results, k):
     patk = len([i for i in Q_results[0:k] if i in GT])/k
     return patk

#this function returns the mean P@k of a list of query results giving their ground truth and k
def mean_P_at_k(groud_truth_dict, se_results_dict, k):
    patk_list = [] #List of all P@k for all the queries  
    
    for i in se_results_dict.keys(): #for each iteration, i will contain the query_id from the ground truth list which is the same as the ones from the query results list
        GT_qi= groud_truth_dict[i] #Ground truth list for the current query
        Qi_results = se_results_dict[i] #Results list for the current query
        
        patk_i = P_at_k(GT_qi, Qi_results, k)
        
        patk_list.append(patk_i)
    
    return np.mean(patk_list)

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
    
    for i in se_results_dict.keys(): #for each iteration, i will contain the query_id from the ground truth list which is the same as the ones from the query results list
        GT_qi= groud_truth_dict[i] #Ground truth list for the current query
        Qi_results = se_results_dict[i] #Results list for the current query
        
        ndcg_i = ndcg(GT_qi, Qi_results, k)
        
        ndcg_list.append(ndcg_i)
    
    return np.mean(ndcg_list)
#%%
'''PART 1: Search-Engine Evaluation'''
#Part1_1: we want to evaluate 3 search engine using different evaluation measures: P@K, R-Precision, MRR and nDCG
if __name__ == "__main__":
    ground_truth_path = '../dataset/part_1_1/part_1_1__Ground_Truth.tsv'
    Results_SE_1_path = '../dataset/part_1_1/part_1_1__Results_SE_1.tsv'
    Results_SE_2_path = '../dataset/part_1_1/part_1_1__Results_SE_2.tsv'
    Results_SE_3_path = '../dataset/part_1_1/part_1_1__Results_SE_3.tsv'
    
    gt_part1 = fromCsvToDict(ground_truth_path, 'Query_id', 'Relevant_Doc_id')
    se1_part1 = fromCsvToDict(Results_SE_1_path, 'Query_ID', 'Doc_ID')
    se2_part1 = fromCsvToDict(Results_SE_2_path, 'Query_ID', 'Doc_ID')
    se3_part1 = fromCsvToDict(Results_SE_3_path, 'Query_ID', 'Doc_ID')
    
    '''
    all the above variables are dictionary. their structure is the following:
        gt_part1: keys -> Query_id's, values -> list of relevant doc_id's for that query
        sei_part1: keys -> Qury_id's, values -> list of doc_id's returned by the query order by their rank in ascending order
    '''
#%%
    '''P@K tabble'''
    p_at_k_table = np.array([['Search Engine','Mean(P@1)','Mean(P@3)', 'Mean(P@5)', 'Mean(P@10)'],
                    ['SE_1', 
                     mean_P_at_k(gt_part1, se1_part1, 1), 
                     mean_P_at_k(gt_part1, se1_part1, 3), 
                     mean_P_at_k(gt_part1, se1_part1, 5), 
                     mean_P_at_k(gt_part1, se1_part1, 10)],
                    ['SE_2', 
                     mean_P_at_k(gt_part1, se2_part1, 1), 
                     mean_P_at_k(gt_part1, se2_part1, 3), 
                     mean_P_at_k(gt_part1, se2_part1, 5), 
                     mean_P_at_k(gt_part1, se2_part1, 10)],
                    ['SE_3', 
                     mean_P_at_k(gt_part1, se3_part1, 1), 
                     mean_P_at_k(gt_part1, se3_part1, 3), 
                     mean_P_at_k(gt_part1, se3_part1, 5), 
                     mean_P_at_k(gt_part1, se3_part1, 10)]])    

    df = pd.DataFrame(p_at_k_table)
    df.to_csv('../output/p_at_k.csv', header=False, index=False)
#%%
    '''R-Precision table'''
    R_precision_table = np.array([['Search Engine',
                                   'Mean(R-Precision_Distrbution)',
                                   'min(R-Precision_Distrbution)', 
                                   '1st_quartile(R-Precision_Distrbution)', 
                                   'MEDIAN(R-Precision_Distrbution)', 
                                   '3rd_quartile(R-Precision_Distrbution)', 
                                   'MAX(R-Precision_Distrbution)'],
                    ['SE_1', 
                     np.mean(get_r_precision_list(gt_part1, se1_part1)), 
                     np.min(get_r_precision_list(gt_part1, se1_part1)), 
                     np.percentile(get_r_precision_list(gt_part1, se1_part1), 25), 
                     np.median(get_r_precision_list(gt_part1, se1_part1)), 
                     np.percentile(get_r_precision_list(gt_part1, se1_part1), 75),
                     np.max(get_r_precision_list(gt_part1, se1_part1))],
                    ['SE_2', 
                     np.mean(get_r_precision_list(gt_part1, se2_part1)), 
                     np.min(get_r_precision_list(gt_part1, se2_part1)), 
                     np.percentile(get_r_precision_list(gt_part1, se2_part1), 25), 
                     np.median(get_r_precision_list(gt_part1, se2_part1)), 
                     np.percentile(get_r_precision_list(gt_part1, se2_part1), 75),
                     np.max(get_r_precision_list(gt_part1, se2_part1))],
                    ['SE_3', 
                     np.mean(get_r_precision_list(gt_part1, se3_part1)), 
                     np.min(get_r_precision_list(gt_part1, se3_part1)), 
                     np.percentile(get_r_precision_list(gt_part1, se3_part1), 25), 
                     np.median(get_r_precision_list(gt_part1, se3_part1)), 
                     np.percentile(get_r_precision_list(gt_part1, se3_part1), 75),
                     np.max(get_r_precision_list(gt_part1, se3_part1))]])    
    
    df = pd.DataFrame(R_precision_table)
    df.to_csv('../output/R-Precision.csv', header=False, index=False)
#%%
    '''MRR(Mean Reciprocal Rank) output data'''
    MRR_table = np.array([['Search Engine', 'MRR'],
                    ['SE_1', mrr(gt_part1, se1_part1)],
                    ['SE_2', mrr(gt_part1, se2_part1)],
                    ['SE_3', mrr(gt_part1, se3_part1)]])
    
    df = pd.DataFrame(MRR_table)
    df.to_csv('../output/MRR.csv', header=False, index=False) 
#%%
    '''nDCG(normalized Discounted Cumulative Gain) table'''
    nDCG_table = np.array([['Search Engine','Mean(nDCG@1)','Mean(nDCG@3)', 'Mean(nDCG@5)', 'Mean(nDCG@10)'],
                    ['SE_1', 
                     mean_ndcg_at_k(gt_part1, se1_part1, 1), 
                     mean_ndcg_at_k(gt_part1, se1_part1, 3), 
                     mean_ndcg_at_k(gt_part1, se1_part1, 5), 
                     mean_ndcg_at_k(gt_part1, se1_part1, 10)],
                    ['SE_2', 
                     mean_ndcg_at_k(gt_part1, se2_part1, 1), 
                     mean_ndcg_at_k(gt_part1, se2_part1, 3), 
                     mean_ndcg_at_k(gt_part1, se2_part1, 5), 
                     mean_ndcg_at_k(gt_part1, se2_part1, 10)],
                    ['SE_3', 
                     mean_ndcg_at_k(gt_part1, se3_part1, 1), 
                     mean_ndcg_at_k(gt_part1, se3_part1, 3), 
                     mean_ndcg_at_k(gt_part1, se3_part1, 5), 
                    mean_ndcg_at_k(gt_part1, se3_part1, 10)]])    

    df = pd.DataFrame(nDCG_table)
    df.to_csv('../output/nDCG.csv', header=False, index=False)
#%%
    
    
    
    
    
    
    
    
    
    
    
    
    
    