import arff as arff
import numpy as np

datasetRepo = 'BreastCancer'
datasetName = 'breast-cancer-wisconsinBinary'

data_list=[]

for row in arff.load('/home/lukas/Uni/AAThesis/Datasets/'+ datasetRepo +'/'+ datasetName +'.arff'):
    data_list.append(row)

data_np=np.array(data_list,dtype=float)

#eliminate missing

#if you need to cut the first feature e.g. it is the instance number uncomment the next line
#data_np = data_np[:,1:] 

data_Y=data_np[:,-1]

num_insts=data_np.shape[0]

num_pos_examples=int(np.sum(data_Y))
num_neg_examples=num_insts-num_pos_examples
print('Dataset '+datasetName+' is getting Imported...')
print('Number of instances:',num_insts)
print('Number of positive Examples:',num_pos_examples)
print('Number of negative Examples:',num_neg_examples)
np.savetxt('/home/lukas/Uni/AAThesis/DeepRED/data/'+datasetName+'.csv', data_np, delimiter=",",fmt='%d')
print('Import done!')