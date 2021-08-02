# Uses the k-means clustering algorithm to select a maximum of n (initial_k) 
# number of thresholds to divide the activation range of a neuron.

import numpy as np
from obj_data_set import DataSet
import load_restore as lr
from scipy.cluster import vq
import time
import warnings


def accuracy(x, one_hot_y, hypothesis):
	'''
	Calculates the current accuracy
	'''
	number_examples = len(x)
	correct = 0
	for e in range(number_examples):
		prediction = list(hypothesis[e])
		n1 = one_hot_y[e].index(max(one_hot_y[e]))
		n2 = prediction.index(max(prediction))
		if n1 == n2:
			correct += 1
	return float(correct)/float(number_examples)

def k_means_no_error(values, k):
	''' 
	Performs the k-means algorithm
	'''
	prior_idx = None
	not_found = 0
	found = False
	res, idx = vq.kmeans2(values, k)
	return res, idx

def accuracy_with_means(x, y_one_hot, layers, means, weights, bias):
	'''
	Calculates the current accuracy of the network when replacing each
		activation by its closest cluster mean
	'''
	activations = [None]*(layers-1)
	activations[0] = np.tanh(np.matmul(replace_examples_with_mean(x, means[0]), weights[0]) + bias[0])
	for j in range(1, layers-1):
		new_activations = replace_examples_with_mean(activations[j-1], means[j])
		activations[j] = np.tanh(np.matmul(new_activations, weights[j]) + bias[j])
	new_activations = replace_examples_with_mean(activations[layers-2], means[layers-1])
	logits = np.matmul(new_activations, weights[layers-1]) + bias[layers-1]
	return accuracy(x, y_one_hot, logits)

def replace_with_mean(value, neuron_means):
	'''
	Returns the closest mean of a value
	'''
	return min(neuron_means, key=lambda m:abs(m-value))

def replace_examples_with_mean(values, layer_means):
	'''
	Replaces each activation value by its closest cluster mean
	'''
	return [np.array([replace_with_mean(e[i], layer_means[i]) for i in range(len(layer_means))]) for e in values]

def evaluate_means(experiment, split, initial_k):
	'''
	Evaluates the accuracy of the network using the calculated means.
	'''
	dataset_name, model_name, hidden_nodes, softmax, train_folds, train, test, validation, class_dominance, min_set_size, dis_config, rft_pruning_config, rep_pruning_config, target_class_index = lr.get_metadata(experiment)
	train, test, train_folds = lr.load_indexes(split, dataset_name)
	act_train, act_test, weights, bias = lr.load_act_values_paramaters(model_name, train_folds)

	data = DataSet(dataset_name, hidden_nodes)
	data.set_split(train, [], test)
	data.set_act_values(act_train, [], act_test)
	x_train, y_train = data.get_train_x_y()
	x_test, y_test = data.get_test_x_y()
	layers = len(hidden_nodes)+1
	
	means = lr.load_cluster_means(experiment, initial_k, split)
	cut_points = lr.load_cut_points(experiment, initial_k, split)
	
	print(cut_points)
	
	train_accuracy = accuracy_with_means(x_train, y_train, layers, means, weights, bias)
	test_accuracy = accuracy_with_means(x_test, y_test, layers, means, weights, bias)
	print('Train accuracy: ', train_accuracy)
	print('Test accuracy: ', test_accuracy)

def extract_new_means(experiment, split, initial_k):
	'''
	It extracts a new list of means to separate the activation range of each neuron.
	At most initial_k values are selected per neuron, but this number is reduces as long 
	as the accuracy of the network does not decrease from that using initial_k.
	'''
	dataset_name, model_name, hidden_nodes, softmax, train_folds, train, test, validation, class_dominance, min_set_size, dis_config, rft_pruning_config, rep_pruning_config, target_class_index = lr.get_metadata(experiment)
	train, test, train_folds = lr.load_indexes(split, dataset_name)
	act_train, act_test, weights, bias = lr.load_act_values_paramaters(model_name, train_folds)
	data = DataSet(dataset_name, hidden_nodes)
	data.set_split(train, [], test)
	data.set_act_values(act_train, [], act_test)
	x_train, y_train = data.get_train_x_y()
	x_test, y_test = data.get_test_x_y()
	layers = len(hidden_nodes)+1
	input_size = len(x_train[0])
	# 'means' stores a list of lists, where each list is a neuron and the 
	# entries are the cluster means. It is initialized to all the unique values for each neuron
	# Layer 0 referrs to the inputs
	means = [None] * layers
	cut_points = [None] * layers

	k = initial_k
	#means[0] = [vq.kmeans(list(set(e[i] for e in x_train)), k, iter=100)[0] for i in range(input_size)]
	means[0] = [None] * input_size
	cut_points[0] = [None] * input_size
	for n in range(input_size):
		values = [e[n] for e in x_train]
		means[0][n], belonging = k_means_no_error(values, k)
		means_initial = means[0][n].copy()
		enum = [(x,i) for (i,x) in enumerate(means_initial)]
		sorted_enum = [i for (x,i) in sorted(enum)]
		last_belonging = [sorted_enum.index(x) for x in belonging]
		clusters = [[x for (i,x) in enumerate(values) if last_belonging[i] == c] for c in range(len(means_initial))]
		clusters = [c for c in clusters if c]
		mid_points = [(max(clusters[i]) + min(clusters[i+1]))/2 for i in range(len(clusters)-1)]
		if mid_points:
			cut_points[0][n] = mid_points
		else:
			cut_points[0][n] = [0.5]
		
	for l in range(1, layers):
		means[l] = [None] * hidden_nodes[l-1]
		cut_points[l] = [None] * hidden_nodes[l-1]
		for n in range(hidden_nodes[l-1]):
			#means[l] = [vq.kmeans(list(set(e[i] for e in act_train[l-1])), k, iter=100)[0] for i in range(hidden_nodes[l-1])]
			values = [e[n] for e in act_train[l-1]]
			means[l][n], belonging = k_means_no_error(values, k)
			means_initial = means[l][n].copy()
			enum = [(x,i) for (i,x) in enumerate(means_initial)]
			sorted_enum = [i for (x,i) in sorted(enum)]
			last_belonging = [sorted_enum.index(x) for x in belonging]
			clusters = [[x for (i,x) in enumerate(values) if last_belonging[i] == c] for c in range(len(means_initial))]
			clusters = [c for c in clusters if c]
			mid_points = [(max(clusters[i]) + min(clusters[i+1]))/2 for i in range(len(clusters)-1)]
			if mid_points:
				cut_points[l][n] = mid_points
			else:
				cut_points[l][n] = [0]
	
	initial_clustered_accuracy = accuracy_with_means(x_train, y_train, layers, means, weights, bias)

	
	boundaries = []
	for l in range(layers-1, 0, -1):
		print('l', l)
		for n in range(len(means[l])):
			#print('n', n)
			k_current = k
			new_accuracy = 1
			new_means = means[l][n]
			belonging = None
			
			values = [e[n] for e in act_train[l-1]]
			while(k_current > 1 and new_accuracy >= initial_clustered_accuracy):
				last_means = new_means.copy()
				last_belonging = belonging
				k_current -= 1
				#print('k_current', k_current)
				new_means, belonging = k_means_no_error(values, k_current)
				means[l][n] = new_means
				new_accuracy = accuracy_with_means(x_train, y_train, layers, means, weights, bias)
				#print('new_accuracy', new_accuracy)
			if k_current < initial_k-1:
				means[l][n] = last_means	
				enum = [(x,i) for (i,x) in enumerate(last_means)]
				sorted_enum = [i for (x,i) in sorted(enum)]
				last_belonging = [sorted_enum.index(x) for x in last_belonging]
				clusters = [[x for (i,x) in enumerate(values) if last_belonging[i] == c] for c in range(len(last_means))]
				clusters = [c for c in clusters if c]
				mid_points = [(max(clusters[i]) + min(clusters[i+1]))/2 for i in range(len(clusters)-1)]
				if mid_points:
					cut_points[l][n] = mid_points
	print('l', 0)
	for n in range(input_size):
		#print('n', n)
		k_current = k
		new_accuracy = 1
		new_means = means[0][n]
		belonging = None
		values = [e[n] for e in x_train]
		while(k_current > 1 and new_accuracy >= initial_clustered_accuracy):
			last_means = new_means.copy()
			last_belonging = belonging
			k_current -= 1
			new_means, belonging = k_means_no_error(values, k_current)
			means[0][n] = new_means
			new_accuracy = accuracy_with_means(x_train, y_train, layers, means, weights, bias)
		if k_current < initial_k - 1:
			means[0][n] = last_means
			enum = [(x,i) for (i,x) in enumerate(last_means)]
			sorted_enum = [i for (x,i) in sorted(enum)]
			last_belonging = [sorted_enum.index(x) for x in last_belonging]
			clusters = [[x for (i,x) in enumerate(values) if last_belonging[i] == c] for c in range(len(last_means))]
			clusters = [c for c in clusters if c]
			mid_points = [(max(clusters[i]) + min(clusters[i+1]))/2 for i in range(len(clusters)-1)]
			if mid_points:
				cut_points[0][n] = [0.5]
			else:
				cut_points[0][n] = mid_points
	lr.save_cluster_means(means, experiment, k, split)
	lr.save_cut_points(cut_points, experiment, k, split)

