#%%
import numpy as np
import pandas as pd
import math
import hwmodule as hwm
#%%
'''PART 1: Search-Engine Evaluation'''
#Part1_1: we want to evaluate 3 search engine using different evaluation measures: P@K, R-Precision, MRR and nDCG
if __name__ == "__main__":
    ground_truth_path = '../dataset/part_1_1/part_1_1__Ground_Truth.tsv'
    Results_SE_1_path = '../dataset/part_1_1/part_1_1__Results_SE_1.tsv'
    Results_SE_2_path = '../dataset/part_1_1/part_1_1__Results_SE_2.tsv'
    Results_SE_3_path = '../dataset/part_1_1/part_1_1__Results_SE_3.tsv'
    
    gt_part1 = hwm.fromCsvToDict(ground_truth_path, 'Query_id', 'Relevant_Doc_id')
    se1_part1 = hwm.fromCsvToDict(Results_SE_1_path, 'Query_ID', 'Doc_ID')
    se2_part1 = hwm.fromCsvToDict(Results_SE_2_path, 'Query_ID', 'Doc_ID')
    se3_part1 = hwm.fromCsvToDict(Results_SE_3_path, 'Query_ID', 'Doc_ID')
    
    '''
    all the above variables are dictionary. their structure is the following:
        gt_part1: keys -> Query_id's, values -> list of relevant doc_id's for that query
        sei_part1: keys -> Qury_id's, values -> list of doc_id's returned by the query order by their rank in ascending order
    '''
#%%
    '''P@K tabble'''
    p_at_k_table = np.array([['Search Engine','Mean(P@1)','Mean(P@3)', 'Mean(P@5)', 'Mean(P@10)'],
                    ['SE_1', 
                     hwm.mean_P_at_k(gt_part1, se1_part1, 1), 
                     hwm.mean_P_at_k(gt_part1, se1_part1, 3), 
                     hwm.mean_P_at_k(gt_part1, se1_part1, 5), 
                     hwm.mean_P_at_k(gt_part1, se1_part1, 10)],
                    ['SE_2', 
                     hwm.mean_P_at_k(gt_part1, se2_part1, 1), 
                     hwm.mean_P_at_k(gt_part1, se2_part1, 3), 
                     hwm.mean_P_at_k(gt_part1, se2_part1, 5), 
                     hwm.mean_P_at_k(gt_part1, se2_part1, 10)],
                    ['SE_3', 
                     hwm.mean_P_at_k(gt_part1, se3_part1, 1), 
                     hwm.mean_P_at_k(gt_part1, se3_part1, 3), 
                     hwm.mean_P_at_k(gt_part1, se3_part1, 5), 
                     hwm.mean_P_at_k(gt_part1, se3_part1, 10)]])    

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
                     np.mean(hwm.get_r_precision_list(gt_part1, se1_part1)), 
                     np.min(hwm.get_r_precision_list(gt_part1, se1_part1)), 
                     np.percentile(hwm.get_r_precision_list(gt_part1, se1_part1), 25), 
                     np.median(hwm.get_r_precision_list(gt_part1, se1_part1)), 
                     np.percentile(hwm.get_r_precision_list(gt_part1, se1_part1), 75),
                     np.max(hwm.get_r_precision_list(gt_part1, se1_part1))],
                    ['SE_2', 
                     np.mean(hwm.get_r_precision_list(gt_part1, se2_part1)), 
                     np.min(hwm.get_r_precision_list(gt_part1, se2_part1)), 
                     np.percentile(hwm.get_r_precision_list(gt_part1, se2_part1), 25), 
                     np.median(hwm.get_r_precision_list(gt_part1, se2_part1)), 
                     np.percentile(hwm.get_r_precision_list(gt_part1, se2_part1), 75),
                     np.max(hwm.get_r_precision_list(gt_part1, se2_part1))],
                    ['SE_3', 
                     np.mean(hwm.get_r_precision_list(gt_part1, se3_part1)), 
                     np.min(hwm.get_r_precision_list(gt_part1, se3_part1)), 
                     np.percentile(hwm.get_r_precision_list(gt_part1, se3_part1), 25), 
                     np.median(hwm.get_r_precision_list(gt_part1, se3_part1)), 
                     np.percentile(hwm.get_r_precision_list(gt_part1, se3_part1), 75),
                     np.max(hwm.get_r_precision_list(gt_part1, se3_part1))]])    
    
    df = pd.DataFrame(R_precision_table)
    df.to_csv('../output/R-Precision.csv', header=False, index=False)
#%%
    '''MRR(Mean Reciprocal Rank) output data'''
    MRR_table = np.array([['Search Engine', 'MRR'],
                    ['SE_1', hwm.mrr(gt_part1, se1_part1)],
                    ['SE_2', hwm.mrr(gt_part1, se2_part1)],
                    ['SE_3', hwm.mrr(gt_part1, se3_part1)]])
    
    df = pd.DataFrame(MRR_table)
    df.to_csv('../output/MRR.csv', header=False, index=False) 
#%%
    '''nDCG(normalized Discounted Cumulative Gain) table'''
    nDCG_table = np.array([['Search Engine','Mean(nDCG@1)','Mean(nDCG@3)', 'Mean(nDCG@5)', 'Mean(nDCG@10)'],
                    ['SE_1', 
                     hwm.mean_ndcg_at_k(gt_part1, se1_part1, 1), 
                     hwm.mean_ndcg_at_k(gt_part1, se1_part1, 3), 
                     hwm.mean_ndcg_at_k(gt_part1, se1_part1, 5), 
                     hwm.mean_ndcg_at_k(gt_part1, se1_part1, 10)],
                    ['SE_2', 
                     hwm.mean_ndcg_at_k(gt_part1, se2_part1, 1), 
                     hwm.mean_ndcg_at_k(gt_part1, se2_part1, 3), 
                     hwm.mean_ndcg_at_k(gt_part1, se2_part1, 5), 
                     hwm.mean_ndcg_at_k(gt_part1, se2_part1, 10)],
                    ['SE_3', 
                     hwm.mean_ndcg_at_k(gt_part1, se3_part1, 1), 
                     hwm.mean_ndcg_at_k(gt_part1, se3_part1, 3), 
                     hwm.mean_ndcg_at_k(gt_part1, se3_part1, 5), 
                     hwm.mean_ndcg_at_k(gt_part1, se3_part1, 10)]])    

    df = pd.DataFrame(nDCG_table)
    df.to_csv('../output/nDCG.csv', header=False, index=False)
#%%
##Part1_2: he we try to select the best one among 3 search engine
    
    
    
    
    
    
    
    
    
    
    
    
    
    