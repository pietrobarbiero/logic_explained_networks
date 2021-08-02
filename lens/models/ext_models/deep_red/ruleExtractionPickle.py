import pickle

dataset_name = 'breast-cancer-wisconsinBinary'
split_name = '1'

name = dataset_name + '_' + split_name

def showBio(name):
    with open('obj/bio_' + name + '.pkl','rb') as f:
        return pickle.load(f)

def showBNN(name):
    with open('obj/BNN_' + name + '.pkl','rb') as f:
        return pickle.load(f)

def showEx(name):
    with open('obj/example_cond_dict_' + name + '.pkl','rb') as f:
        return pickle.load(f)

def showIndex(name):
    with open('obj/index_list_' + name + '.pkl','rb') as f:
        return pickle.load(f)

def showIndexTrain(name):
    with open('indexes/' + name + '_train.pkl','rb') as f:
        return pickle.load(f)

def showIndexTest(name):
    with open('indexes/' + name + '_test.pkl','rb') as f:
        return pickle.load(f)

#print('Datatype IndexTrain: ' + str(type(showIndexTrain(name))))
#print('Datatype IndexTest: ' + str(type(showIndexTest(name))))
print('')

def showingRaw(showAllBio=True, showAllBNN=True, showAllEx=False, showAllInd=False, showAllTrainInd=False, showAllTestInd=False):
    if showAllBio: print('Datatype Bio: ' + str(type(showBio(name))));print(len(showBio(name)));print('Bio:');print(showBio(name))
    if showAllBNN: print('Datatype BNN: ' + str(type(showBNN(name))));print(len(showBNN(name)));print('BNN:');print(showBNN(name))
    if showAllEx: print('Datatype Ex: ' + str(type(showEx(name))));print(len(showEx(name)));print('Ex:');print(showEx(name))
    if showAllInd: print('Datatype Index: ' + str(type(showIndex(name))));print(len(showIndex(name)));print('Index:');print(showIndex(name))
    if showAllTrainInd: print('Datatype IndexTrain: ' + str(type(showIndexTrain(name))));print(len(showIndexTrain(name)));print('IndexTrain:');print(showIndexTrain(name))
    if showAllTestInd: print('Datatype IndexTest: ' + str(type(showIndexTest(name))));print(len(showIndexTest(name)));print('IndexTest:');print(showIndexTest(name))

"""
print(len(showIndexTrain(name)))
print(len(showIndexTest(name)))

print('IndexTrain:')
print(showIndexTrain(name))
print('IndexTest:')
print(showIndexTest(name))
"""
showingRaw()