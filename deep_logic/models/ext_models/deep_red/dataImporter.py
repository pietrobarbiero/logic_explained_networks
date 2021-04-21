import numpy as np
import csv

datasetRepo = 'InternetAd'
datasetName = 'ad'

list_of_list = []

with open('/home/lukas/Uni/AAThesis/Datasets/'+ datasetRepo +'/'+ datasetName +'.data', 'r') as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(',')]
        list_of_list.append(inner_list)

print(list_of_list)

np.savetxt('/home/lukas/Uni/AAThesis/DeepRED/data/'+datasetName+'.csv', list_of_list, delimiter=",",fmt='%d')