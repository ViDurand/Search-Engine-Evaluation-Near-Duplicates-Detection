from bs4 import BeautifulSoup
import sys
import string
import os.path
import os
import re
import random
import time
from tqdm import tqdm
import binascii
import math
import csv

# script part 2.1
# lyrics collection directory
path = '../dataset/part_2_1/lyrics_collection__CONVERTED'
shingle_size = 3


set_id= []
#lyrics full dataset
#grabbing the file name which we will use as ID in our input file 
for file in tqdm(os.listdir(path)):
    if file.endswith(".html"):
        set_id.append(file)

shingledLyric = {} #store the lyric shingles here
shingles_db = [] #store all the shingles in the documents here
#in here we will read each file and convert the lyrics in sets of shingle        
for file_name in tqdm(set_id):
    try:
        #open the files
        filename= os.path.join(path, file_name)
        f = open(filename, 'r')
        soup = BeautifulSoup(f.read(), 'html.parser') #read the html

        #grab and format the body of the lyrics file
        body = re.sub('(<.*?>)|(\")|(\W)', ' ', str(soup.body)).lower()
        #split the files in workds
        words = body.strip().split()
        #initialize our shingles variable
        shingles = []
        #create the shingles
        for i in range(len(words) - shingle_size+1):
            shingles.append(" ".join(words[i:i+shingle_size]))
            
        #save the shingles in the document's shingles variable
        if len(shingles) > 0:
            shingledLyric[file_name] =  shingles
        #save the shingle in our total shingle set for the dataset
            shingles_db.extend(shingles)
    except:
        print('Error file #', file_name)
        

#check the size of our shigle set for processing the hash functions
print(len(shingles_db), len(set(shingles_db)))
shingles_set = {}
#create natural numbers for all the shingle we have with the hashing function crc 32
for shingle in tqdm(set(shingles_db)):
    shingles_set[shingle] = binascii.crc32(shingle.encode('utf8')) & 0xffffffff
    

single_db = [] #empty it to save some RAM 
        
        
################################################
num_hash_functions = 300
upper_bound_on_number_of_distinct_terms  = 6962837
#upper_bound_on_number_of_distinct_terms =   138492
#upper_bound_on_number_of_distinct_terms =  3746518

################################################


### primality checker
def is_prime(number):
	for j in range(2, int(math.sqrt(number)+1)):
		if (number % j) == 0: 
			return False
	return True

sketch_values=[]
#generating the different hash values
print( "a	b	p	n")
for hash_function_id in tqdm(range(num_hash_functions)):
    a = random.randint(1, upper_bound_on_number_of_distinct_terms-1)
    b = random.randint(0, upper_bound_on_number_of_distinct_terms-1)
    p = random.randint(upper_bound_on_number_of_distinct_terms, 10*upper_bound_on_number_of_distinct_terms)
    while is_prime(p) == False:
        p = random.randint(upper_bound_on_number_of_distinct_terms, 10*upper_bound_on_number_of_distinct_terms)
    sketch_values.append([a,b,p,upper_bound_on_number_of_distinct_terms ]) 

#saving our minhashed dict which we will use with the tools
input_dict={}
for k, v in tqdm(shingledLyric.items()):
    minhash = []
    for a, b, p, n in sketch_values:
        h_values = [(a * shingles_set[shingle] + b % p) % n for shingle in v]
        minhash.append(min(h_values))
    input_dict[k] = minhash
    


#saving the values in our input files
with open('../output/lyrics_full_hashed.tsv', 'w') as output_file:
    for k, v in input_dict.items():
        output_file.write(k + "\t" + str(v) + "\n")

with open ('../output/300_hash_functions.tsv', 'w') as output_file:
    for a,b,p,n in sketch_values:
        output_file.write(str(a) + "\t" + str(b) + "\t" + str(p) + "\t" + str(n) + "\n" )
        
'''

#script to find near duplicate: should be run from comand prompt
java -Xmx1G tools.NearDuplicatesDetector lsh_plus_min_hashing 0.85 23 13 ../output/300_hash_functions.csv ./lyrics_full_hashed.tsv ../output/part221_dmthw1_results_300Hash_lsh_085_23_13.csv

#script to find near duplicate candidates: should be run from comand prompt
java -Xmx1G tools.NearDuplicatesDetector lsh 23 13 ./300_hash_functions.tsv ../output/lyrics_full_hashed.csv ../output/part221_dmthw1_results_300Hash_lsh_085_23_13_CANDIDATES.csv

'''