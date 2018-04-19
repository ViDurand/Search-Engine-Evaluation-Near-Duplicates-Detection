#%%
import numpy as np
import pandas as pd
import hwmodule as hwm
import ast
import sys

#%%
print("***Part2_2a: “Set-Size-Estimation problem”***")

#this function compute the sketches_list of the union
def sketch_of_union(union_set, sketches_dict):
    union_set = list(union_set)
    sketch_of_union = sketches_dict[union_set[0]]
    for sketchID in union_set[1:]:
        actual_sketch = sketches_dict[sketchID]
        for i in range(len(actual_sketch)):
            if actual_sketch[i] < sketch_of_union[i]:
                sketch_of_union[i] = actual_sketch[i]
    
    return sketch_of_union


if __name__ == "__main__":
    min_hash_sketches_path = '../dataset/part_2_2/HW_1_part_2_2_dataset__min_hash_sketches.tsv'
    sets_ids_for_union = '../dataset/part_2_2/HW_1_part_2_2_dataset__SETS_IDS_for_UNION.tsv'
    #min_hash_sketches_path = sys.argv[1]
    #sets_ids_for_union = sys.argv[2]
    
    #reading all the sketches lists and storing them into a dictionary
    df = pd.read_csv(min_hash_sketches_path, sep='\t', header=0)
    keys = list(df['Min_Hash_Sketch_INTEGER_Id'])
    values = list(df['Min_Hash_Sketch'])
    values = [ast.literal_eval(e) for e in values]
    min_hash_sketches = dict(zip(keys, values))
    
    #reading all the set_of_sketchs and storing them into a dictionary
    df = pd.read_csv(sets_ids_for_union, sep='\t', header=0)
    
    pd.options.mode.chained_assignment = None  # default='warn'
    
    df['Min_Hash_Sketch_Union'] = df['set_of_sets_ids']
    
    i = 0
    for s in df['set_of_sets_ids']:
        df['Min_Hash_Sketch_Union'][i] = sketch_of_union(ast.literal_eval(s), min_hash_sketches)
        i=i+1
 

'''
    #compute all the estimated sizes ad saving them into a dictionary
    universe_size = 1123581321
    estimated_set_size_dict = dict()
    for sketchID in min_hash_sketches.keys():
        estimated_set_size_dict[sketchID] = hwm.set_Size_Estimator(min_hash_sketches[sketchID], universe_size)
    
    
    #saving the estimated sizes into a csv file
    estimated_sizes_df = pd.DataFrame(list(estimated_set_size_dict.items()), columns=['Min_Hash_Sketch_INTEGER_Id', 'ESTIMATED_ORIGINAL_SET_SIZE'])
    estimated_sizes_df.to_csv('../output/OUTPUT_HW_1_part_2_2_a.csv', header=False, index=False)
#%%
print("***Work compleated, check the output directory for output file***")
#%%
'''