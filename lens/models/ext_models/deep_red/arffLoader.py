import arff as arff
import numpy as np
import sklearn.model_selection as ms
import pickle

"""
@attribute 'Temperature>37,3C/Schuettelfrost' {0,1}
@attribute (Reiz)Husten {0,1}
@attribute Kontakt {0,1}
@attribute Anosmie/Ageusie {0,1}
@attribute 'Abgeschlagenheit' {0,1}
@attribute Gliederschmerzen {0,1}
@attribute Halsschmerzen {0,1}
@attribute Kopfschmerzen {0,1}
@attribute 'unspezifischeAbdominal-schmerz/Diarrhoe/Erbrechen' {0,1}
@attribute Schnupfen {0,1}
@attribute 'RespiratorischeSymptomatik' {0,1}
@attribute 'SARS-CoV-2-PCRpositiv' {0,1}
"""

oversampling= 50

datasetName = 'C'
splitName = '1'

data_arff = arff.load('data/'+ datasetName +'.arff')
data_list=[]

for row in arff.load('data/'+ datasetName +'.arff'):
    data_list.append(row)

data_np=np.array(data_list,dtype=float)
#data_np=data_np[0:10]
print(data_np)

#remove missings hack
temp = np.isfinite(data_np)
data_np[~temp]=-1


data_X=data_np[:,0:-1]
data_Y=data_np[:,-1]

num_insts=data_np.shape[0]

num_pos_examples=int(np.sum(data_Y))
num_neg_examples=num_insts-num_pos_examples

seed=1
num_folds=2
cv = ms.StratifiedKFold(n_splits=num_folds,shuffle=True, random_state=seed)

for fold, (train_idx, test_idx) in enumerate(cv.split(data_X, data_Y)):
    train_data_X=data_X[train_idx]
    train_data_Y=data_Y[train_idx]
    pos_idx=train_data_Y== 1
    positives_X = train_data_X[pos_idx]
    positives_Y = train_data_Y[pos_idx]
    train_data_X = np.vstack([train_data_X, np.repeat(positives_X, repeats=int(oversampling - 1), axis=0)])
    train_data_Y = np.concatenate([train_data_Y, np.repeat(positives_Y, repeats=int(oversampling - 1), axis=0)])

def saveTrainInd():
    with open('indexes/'+datasetName+'_'+splitName+'_train.pkl', 'wb') as f:
            pickle.dump(train_idx,f,pickle.HIGHEST_PROTOCOL)

def saveTestInd():
    with open('indexes/'+datasetName+'_'+splitName+'_test.pkl', 'wb') as f:
		    pickle.dump(test_idx, f, pickle.HIGHEST_PROTOCOL)


train_data_Y = np.array([train_data_Y])
train_data = np.concatenate((train_data_X, train_data_Y.T), axis=1)


def saveDataset():
    np.savetxt('data/'+datasetName+'_'+splitName+'.csv', train_data, delimiter=",",fmt='%d')


#saveTestInd()
#saveTrainInd()
#saveDataset()

pos_examples = np.sum(train_data_Y)
print(pos_examples,train_idx, test_idx)
print("done")