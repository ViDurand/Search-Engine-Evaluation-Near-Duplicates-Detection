#%%
import numpy as np
import pandas as pd
import hwmodule as hwm
import ast
import sys

#%%
print("***Part2_2b: “Unions-Size-Estimation problem”***")

if __name__ == "__main__":
    #min_hash_sketches_path = '../dataset/part_2_2/HW_1_part_2_2_dataset__min_hash_sketches.tsv'
    #sets_ids_for_union = '../dataset/part_2_2/HW_1_part_2_2_dataset__SETS_IDS_for_UNION.tsv'
    min_hash_sketches_path = sys.argv[1]
    sets_ids_for_union = sys.argv[2]
    
    #reading all the sketches lists and storing them into a dictionary
    df = pd.read_csv(min_hash_sketches_path, sep='\t', header=0)
    keys = list(df['Min_Hash_Sketch_INTEGER_Id'])
    values = list(df['Min_Hash_Sketch'])
    values = [ast.literal_eval(e) for e in values]
    min_hash_sketches = dict(zip(keys, values))
    
    #reading all the set_of_sketchs and storing them into a dictionary
    df = pd.read_csv(sets_ids_for_union, sep='\t', header=0)        
    df['Min_Hash_Sketch_Union'] = df['set_of_sets_ids'] #this column will contain the generated Min_Hash_Sketch of the union
    data = df.values #converting the dataframe into numpyarray for easy workout
    
    #compute all the sketch_of_union's and the estimated union sizes at once and subtitute the 3rd column with them
    universe_size = 1123581321
    for row in range(data.shape[0]):
        union_set = list(ast.literal_eval(data[row, 1]))
        data[row, 2] = hwm.set_Size_Estimator(hwm.sketch_of_union(union_set, min_hash_sketches), universe_size)
    
    #saving the estimated sizes into a csv file
    data_df = pd.DataFrame(data, columns=['Union_Set_id','set_of_sets_ids','ESTIMATED_UNION_SIZE'])
    data_df.to_csv('../output/OUTPUT_HW_1_part_2_2_b.csv', header=True, index=False)
#%%
print("***Work compleated, check the output directory for output file***")
#%%
