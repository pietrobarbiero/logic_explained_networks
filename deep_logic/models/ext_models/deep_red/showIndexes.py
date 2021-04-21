import pickle
import arff as arff
import numpy as np

dataset = 'C_P_OS50'
split = '1'

name = dataset + '_' + split

def loadTestEx(name,indlist):
    data_list=[]
    pos_count = 0
    forcount = 0

    for row in arff.load('data/'+ name +'.arff'):
        data_list.append(row)

    data_np=np.array(data_list,dtype=float)
    print(data_np)

    for ind in indlist:
        forcount += 1
        pos_count += data_np[ind,-1]
        
    print(len(data_np))
    print(pos_count)

def showIndex(name):
    with open('obj/index_list_' + name + '.pkl','rb') as f:
        return pickle.load(f)

def showIndexTrain(name):
    with open('indexes/' + name + '_train.pkl','rb') as f:
        return pickle.load(f)

def showIndexRaw(name):
    with open('indexes/' + name + '.pkl','rb') as f:
        return pickle.load(f)

def showIndexTest(name):
    with open('indexes/' + name + '_test.pkl','rb') as f:
        return pickle.load(f)
"""
print('Index:')
#print(showIndex(name))
print('IndexTrain:')
print(showIndexTrain(name))
print('IndexTest:')
print(showIndexTest(name))
"""
testListe = showIndexTest(name)
trainListe = showIndexTrain(name)
#indexListe = showIndexRaw(name)

loadTestEx(dataset,trainListe)
print(len(showIndexTrain(name)))
print(showIndexTrain(name))
