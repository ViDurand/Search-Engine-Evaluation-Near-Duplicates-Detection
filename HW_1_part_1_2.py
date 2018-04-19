#%%
import numpy as np
import pandas as pd
import hwmodule as hwm
import itertools

#%%
print('''***Part1_2: he we try to select the best one among 3 search engine***''')
if __name__ == "__main__":
    print("\nLoading the files...")
    ground_truth_path = '../dataset/part_1_2/part_1_2__Ground_Truth.tsv'
    Results_SE_1_path = '../dataset/part_1_2/part_1_2__Results_SE_1.tsv'
    Results_SE_2_path = '../dataset/part_1_2/part_1_2__Results_SE_2.tsv'
    Results_SE_3_path = '../dataset/part_1_2/part_1_2__Results_SE_3.tsv'
    
    gt_part1_2 = hwm.fromCsvToDict(ground_truth_path, 'Query_id', 'Relevant_Doc_id')
    se1_part1_2 = hwm.fromCsvToDict(Results_SE_1_path, 'Query_ID', 'Doc_ID')
    se2_part1_2= hwm.fromCsvToDict(Results_SE_2_path, 'Query_ID', 'Doc_ID')
    se3_part1_2 = hwm.fromCsvToDict(Results_SE_3_path, 'Query_ID', 'Doc_ID')

#%%
    print('''Some simple stats on data...''')
    Stats_table_2 = np.array([['Dataset', 'Number of Documents', 'Number of Queries'],
                    ['GT', len(set(list(itertools.chain.from_iterable(gt_part1_2.values())))),len(gt_part1_2.keys())],
                    ['SE_1', len(set(list(itertools.chain.from_iterable(se1_part1_2.values())))), len(se1_part1_2.keys())],
                    ['SE_2', len(set(list(itertools.chain.from_iterable(se2_part1_2.values())))), len(se2_part1_2.keys())],
                    ['SE_3', len(set(list(itertools.chain.from_iterable(se3_part1_2.values())))), len(se3_part1_2.keys())]])
    
    df = pd.DataFrame(Stats_table_2)
    df.to_csv('../output/Stats_table_part_1_2.csv', header=False, index=False)

#%%
#The data we have here show that we are dealing with ranking algorithms. each query retrives 200 documents ranked.
#this analys make us understand that Precision, Recall and even their harmonic mean (F-measure) are not good metrics for picking the
#best search engine here.
#However, since the “NoobDataScience” app will return to the user only four of the query results, we assume that the app picks the first 4 results from the
#ranking algorithm and return them in a casual order. And this changes everything. considering only the first query, the 3 search engines will return the following results:
    
    print('SE1-Query1-output: ', set(se1_part1_2[1][:4]))
    print('SE1-Query1-output: ', set(se2_part1_2[1][:4]))
    print('SE1-Query1-output: ', set(se3_part1_2[1][:4]))

#in this particular case, we are dealing with a normal retrievial system without ranking. thus, measures as P@K, MAP, MRR, or nDCG are not useful here since they take into
#account the rank.
#Then, the metric were are going to use here will be the mean(F-measure). we are not simply using mean(P) cause we want to take into account not only the precision but also the reccall.
    
    print('''Mean F_measure output data...''')
    F_measure_table = np.array([['Search Engine', 'mean(F_measure)'],
                    ['SE_1', hwm.mean_F_measure(gt_part1_2, se1_part1_2)],
                    ['SE_2', hwm.mean_F_measure(gt_part1_2, se2_part1_2)],
                    ['SE_3', hwm.mean_F_measure(gt_part1_2, se3_part1_2)]])
    
    df = pd.DataFrame(F_measure_table)
    df.to_csv('../output/F_measure_table_part_1_2.csv', header=False, index=False) 

#The F-mesures tell us that the “NoobDataScience” company should choose the SE 3 since it has the greatest F-measure
#%%
    print('''\n****Work coppleted check the output directory for the csv files***''' )
#%%