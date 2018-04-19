#%%
import numpy as np
import pandas as pd
import hwmodule as hwm
import ast
import sys

#%%
print("***Part2_2a: “Set-Size-Estimation problem”***")

if __name__ == "__main__":
    #min_hash_sketches_path = '../dataset/part_2_2/HW_1_part_2_2_dataset__min_hash_sketches.tsv'
    min_hash_sketches_path = sys.argv[1]
    
    #reading all the sketches lists and storing them into a dictionary
    df = pd.read_csv(min_hash_sketches_path, sep='\t', header=0)
    keys = list(df['Min_Hash_Sketch_INTEGER_Id'])
    values = list(df['Min_Hash_Sketch'])
    values = [ast.literal_eval(e) for e in values]
    min_hash_sketches = dict(zip(keys, values))
    
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