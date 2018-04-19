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


stringTable = str.maketrans({key: None for key in string.punctuation}) #string processing 

path = '../dataset/part_2_1/lyrics_collection/'


lyrics = [] #lyrics collection
lyricsText = {} #lyrics Content

print("Step 1 Loading the files")
print ("Please Be Patient, it will take 15 mins")

rawLyrics = ''

for file in tqdm(os.listdir(path)):
    if file.endswith(".html"):
        filename = os.path.join(path, file)
        f = open(filename, 'r',  encoding='utf8')
        rawLyrics = rawLyrics + f.read() #save all lyrics in one variable

#find the bodies of each song 
soup = BeautifulSoup(rawLyrics, "html.parser")
bodies = soup.findAll('body')
i = 0 #index for the songIDs
for body in tqdm(bodies):
    lyricsText[i] = body
    lyrics.append(
        re.sub(' +', ' ', str(body).replace("<body>", "").replace("</body>", "").translate(stringTable)
               .replace("", "").replace("\n", " ").lower()))
    i = i + 1

print ('Size of lyrics dataset: ' + str(len(lyrics)))


i = 0
data = {} #data will contain as key the songID and as value the lyrics words

t = {}

#assign converted lyrics to each song ids
for lyric in tqdm(lyrics):
    data[i] = lyric
    # split in to words and save words instaed of full text
    data[i] = re.sub("[^\w]", " ", data[i]).split()

    # remove rows with empty values from dictionary d
    if data[i]:
        i = i + 1
    else:
        del data[i]
        del body[i]
        

'''
SHRINGLING PROCESS DOWN BELOW
Shringling the lyrics the set of shingles will be a set of natural numbers.
Before shingling a document, the punctuations have been removed and all words are in lower-case. 
The length of each shingle will be 3.
We will shingle only the lyric of the song.

'''

shringledData = {}

lyricTitles = []

totalShingles = 0
shingleIndex = 0
shingle_size = 3

t0 = time.time()
# loop through all the lyrics
for i in tqdm(range(0, len(data))):

    convertedLyric = data[i]
    docID = i

    # Maintain a list of all document IDs.
    lyricTitles.append(docID)

    # save the various shingles of one song
    shingledLyric = set()
    shingledLyricHashed = set() # save hashed values of the shingles hashed shingles

    shingle = [] #shingles 
    
    for index in range(len(convertedLyric) - shingle_size + 1):

        shingle = convertedLyric[index:index + shingle_size]
        shingle = ' '.join(shingle)

        # Hash the shingle 
        hashShingle = binascii.crc32(shingle.encode('utf8') )

        if shingle not in shingledLyric:
            shingledLyric.add(shingle)

        if hashShingle not in shingledLyricHashed:
            shingledLyricHashed.add(hashShingle)
            shingleIndex = shingleIndex + 1
        else:
            del shingle
            index = index - 1

    # Store the completed list of shingles for this document in the dictionary.
    shringledData[docID] = shingledLyricHashed

totalShingles = shingleIndex