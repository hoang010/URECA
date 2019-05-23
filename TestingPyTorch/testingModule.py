import os
import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT

from os import listdir
from os.path import isfile, join

def give_ids(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    result = []

    for file in onlyfiles:
        if(('.xml' in file) or ('.jpg' in file)):
            if ((path + '/'+file[:-4]) not in result):
                result.append((path + '/'+file[:-4]))
    return result

cur_path = os.getcwd()
path = 'D:/training data'
os.chdir(path)
items = os.listdir()
print("doing ...")
os.chdir(cur_path)
results = []
for x in range (0,len(items)):
    items[x] = "D:/training data/" + items[x]
for item in items:
    results.append(give_ids(item))

file = open('trainval.txt', 'w')

for result in results:
    print(result)
    for item in result: 
        file.write(item + '\n')

file.close()

print("done!")

