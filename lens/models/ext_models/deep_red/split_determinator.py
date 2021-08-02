# Divides the data into train, validation and test sets, or determines the cross-validation splits. 
# The indexes of the splits are stored so experiments can be reproduced.

from . import load_restore as lr
import csv
import random
import math
import sys

def split_train_validation(train, validation, act_values):
	'''
	Receives an array of train indexes, an array of validation indexes which is a subset of the train indexes,
	and the train activation values. It splits the train activation values so that the validation ones are separated
	'''
	train_indexes = [t for t in train if t not in validation]
	validation_indexes = [v for v in train if v not in train_indexes]
	vali_indexes_i = [train.index(v) for v in validation]
	act_values_vali = [[layer[row] for row in range(len(layer)) if row in vali_indexes_i] for layer in act_values]
	act_values_train = [[layer[row] for row in range(len(layer)) if row not in vali_indexes_i] for layer in act_values]
	return train_indexes, validation_indexes, act_values_train, act_values_vali

def hundred_split(dataset):
	file_name = 'data/'+dataset+'.csv'
	with open(file_name,'r') as f:
		data_iter = csv.reader(f, delimiter = ',')
		data = [[float(attr) for attr in data] for data in data_iter]
		l = len(data)
	return range(l), range(l)

	
def split_maintaining_class(dataset, percentage, test_indexes = None):
	'''
	It splits the data in such a way that the class distribution is maintained.
	The train set has the size specified by the 'percentage' parameter.
	If 'test_indexes' is set, the method is being used to get splits to extract rules,
	and the data has already been devided into indexes used to train the network and to test on.
	In this case, only the train indexes are returned.
	'''
	file_name = 'data/'+dataset+'.csv'
	with open(file_name,'r') as f:
		data_iter = csv.reader(f, delimiter = ',')
		data = [[float(attr) for attr in data] for data in data_iter]
	if test_indexes:
		example_indexes = [i for i in range(len(data)) if i not in test_indexes]
	else:
		example_indexes = range(len(data))
	classes = list(set(d[-1] for d in data))
	examples = [[e for e in example_indexes if data[e][-1] == c] for c in classes]
	# Classes with less examples are placed first
	classes_indexes = sorted(range(len(classes)), key=lambda x: len(examples[x]))
	classes = [classes[ci] for ci in classes_indexes]
	examples = [examples[ci] for ci in classes_indexes]
	train = []
	test = []
	for c in examples:
		random.shuffle(c)
		total = len(c)		
		for_train = int(round((float(total) * int(percentage))/100))
		if for_train == 0:
			for_train = 1
		train += c[:for_train]
		test += c[for_train:]
	random.shuffle(train)
	random.shuffle(test)
	if test_indexes:
		return train
	else:
		return train, test

def cv_maintaining_class(dataset, k, train_indexes = None):
	'''
	It returns the indexes devided into k equal groups, while respecting the class distribution.
	'''
	file_name = 'data/'+dataset+'.csv'
	with open(file_name,'r') as f:
		data_iter = csv.reader(f, delimiter = ',')
		data = [[float(attr) for attr in data] for data in data_iter]
	if train_indexes:
		example_indexes = [i for i in range(len(data)) if i in train_indexes]
	else:
		example_indexes = range(len(data))
	classes = list(set(d[-1] for d in data))
	examples = [[e for e in example_indexes if data[e][-1] == c] for c in classes]
	# Classes with less examples are placed first
	classes_indexes = sorted(range(len(classes)), key=lambda x: len(examples[x]))
	classes = [classes[ci] for ci in classes_indexes]
	examples = [examples[ci] for ci in classes_indexes]
	splits = [None] * k
	for j in range(k):
		splits[j] = []
	for c in examples:
		if len(c) < k:
			sys.exit("Error: Not enough examples so that each split has at least one instance of each class")
		else:
			random.shuffle(c)
			pro_chunk = int(len(c)/k)
			split_examples = [c[i:i + pro_chunk] for i in range(0, len(c), pro_chunk)]
			for j in range(k):
				splits[j] += split_examples[j]
			if len(split_examples) > k:
			# If there is a last chunk of the remaining examples, this is devided into those
			# splits that have the least examples
				while split_examples[k]:
					min_length = min(splits, key=lambda x:len(x))
					min_length.append(split_examples[k].pop())
	for s in splits:
		random.shuffle(s)
	return splits

def initial_splits(dataset, split_name, percentage = 50):
	'''
	Determines the splits the network will use to be trained.
	
	param dataset: name of dataset
	param split_name: name that will be assigned to split
	param percentage: percentage of the data that is used for training
	'''
	if dataset == 'xor-8':
		all_indexes = range(256)
		random.shuffle(all_indexes)
		train_indexes = all_indexes[:150]
		test_indexes = all_indexes[150:]
	else:
		train_indexes, test_indexes = split_maintaining_class(dataset, percentage)
	lr.save_train_indexes(train_indexes, dataset, split_name)
	lr.save_test_indexes(test_indexes, dataset, split_name)

def cross_validation_folds(dataset, k, train_vali_splits = [1], return_names=False):
	splits = cv_maintaining_class(dataset, k)
	if len(splits) != k:
		sys.exit("Error: Different splits than k")
	split_names = []
	for i in range(k):
		name_k = 'cv'+str(k)+'-'+str(i)
		lr.save_indexes(splits[i], dataset, name_k)
		split_names.append(name_k)
	if return_names:
		return split_names

def extract_tv_for_100(dataset, train_vali_splits):
	train_indexes, test_indexes, train_folds = lr.load_indexes('100', dataset)
	for s in train_vali_splits:
		vali = cv_maintaining_class(dataset, s, train_indexes)
		for j in range(s):
			name = 'tv'+str(s)+'-'+str(j)
			lr.save_vali_split(vali[j], name, dataset, '100')
				
				
				
				
