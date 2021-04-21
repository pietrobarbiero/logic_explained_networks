# Calculates some measures that evaluates different characteristics of 
# the networks and the extracted models

import numpy as np

def network_accuracy(output_char, data):
	'''
	Determines the accuracy of a network with regards to the output
	characteristic (compared to the class value).

	param output_char -- output characteristic of interest
	param data -- an instance of DataSet
	'''
	correct_predictions_train = 0
	correct_predictions_vali = 0
	correct_predictions_test = 0	
	for i in data.train_indexes + data.vali_indexes + data.test_indexes:
		real_value = data.examples[i].class_value
		network_output = list(data.examples[i].values[data.network_length-1])
		network_value = network_output.index(max(network_output))
		if real_value == network_value:
			if i in data.train_indexes:
				correct_predictions_train += 1
			elif i in data.vali_indexes:
				correct_predictions_vali += 1
			elif i in data.test_indexes:
				correct_predictions_test += 1
	if data.num_train > 0:
		train_accuracy = float(correct_predictions_train) / data.num_train
	else:
		train_accuracy = 1
	if data.num_vali > 0:
		vali_accuracy = float(correct_predictions_vali) / data.num_vali
	else:
		vali_accuracy = 1
	if data.num_test > 0:
		test_accuracy = float(correct_predictions_test) / data.num_test
	else:
		test_accuracy = 1
	return train_accuracy, vali_accuracy, test_accuracy
	
def network_precision(output_char, data):
	'''Determines the precision of a network.
	P = tp/(tp+fp)
	
	param output_char -- output characteristic of interest
	param data -- an instance of DataSet
	'''
	tp_train = 0
	fp_train = 0
	tp_vali = 0
	fp_vali = 0
	tp_test = 0
	fp_test = 0
	for i in data.train_indexes + data.vali_indexes + data.test_indexes:
		real_value = data.examples[i].class_value
		network_value = data.examples[i].values[data.network_length-1][output_char[0]]
		if network_value > output_char[1]:
			if real_value == output_char[0]:
				if i in data.train_indexes:
					tp_train += 1
				elif i in data.vali_indexes:
					tp_vali += 1
				elif i in data.test_indexes:
					tp_test += 1
			else:
				if i in data.train_indexes:
					fp_train += 1
				elif i in data.vali_indexes:
					fp_vali += 1
				elif i in data.test_indexes:
					fp_test += 1
	if tp_train+fp_train > 0:
		train_prec = float(tp_train)/float(tp_train+fp_train)
	else:
		train_prec = 1
	if tp_vali+fp_vali > 0:
		vali_prec = float(tp_vali)/float(tp_vali+fp_vali)
	else:
		vali_prec = 1
	if tp_test+fp_test > 0:
		test_prec = float(tp_test)/float(tp_test+fp_test)
	else:
		test_prec = 1
	return train_prec, vali_prec, test_prec

def network_recall(output_char, data):
	'''
	Determines the recall of a network.
	R = tp/(tp+fn)
	
	param output_char -- output characteristic of interest
	param data -- an instance of DataSet
	'''
	tp_train = 0
	fn_train = 0
	tp_vali = 0
	fn_vali = 0
	tp_test = 0
	fn_test = 0
	for i in data.train_indexes + data.vali_indexes + data.test_indexes:
		real_value = data.examples[i].class_value
		network_value = data.examples[i].values[data.network_length-1][output_char[0]]
		if real_value == output_char[0]:
			if network_value > output_char[1]:
				if i in data.train_indexes:
					tp_train += 1
				elif i in data.vali_indexes:
					tp_vali += 1
				elif i in data.test_indexes:
					tp_test += 1
			else:
				if i in data.train_indexes:
					fn_train += 1
				elif i in data.vali_indexes:
					fn_vali += 1
				elif i in data.test_indexes:
					fn_test += 1
	if tp_train+fn_train > 0:
		train_recall = float(tp_train)/float(tp_train+fn_train)
	else:
		train_recall = 1
	if tp_vali+fn_vali > 0:
		vali_recall = float(tp_vali)/float(tp_vali+fn_vali)
	else:
		vali_recall = 1
	if tp_test+fn_test > 0:
		test_recall = float(tp_test)/float(tp_test+fn_test)
	else:
		test_recall = 1
	return train_recall, vali_recall, test_recall

def porcentace_zero_weights(weights):
	'''
	Determines the percentage of weights that are cero in the network

	param weights -- weight matrixes
	'''
	entry_count = 0
	zero_entries = 0
	for matrix in weights:
		entry_count += matrix.size 
		zero_entries += matrix.size - np.count_nonzero(matrix)
	return float(zero_entries)/entry_count

def porcentage_zero_activations(data, hidden_nodes):
	'''
	Determines the percentage of hidden activations that are cero in the network

	param data -- an instance of DataSet
	param hidden_nodes -- list of lists with the number of neurons on each layer
	'''
	output_layer = data.network_length-1
	zeros = 0
	activations = 0
	for layer in range(1, output_layer):
		for node in range(0, hidden_nodes[layer-1]):
			values = data.get_act_all_examples(layer, node)
			activations += len(values)
			zeros += sum(1 for v in values if v == 0)
	return float(zeros)/float(activations)

def avg_neuron_deviation_from_center(data, hidden_nodes):
	'''
	Determines the deviation of neuron from the mean divided my the number of neurons

	param data -- an instance of DataSet
	param hidden_nodes -- list of lists with the number of neurons on each layer
	'''
	output_layer = data.network_length-1
	deviation = 0
	for layer in range(1, output_layer):
		for node in range(0, hidden_nodes[layer-1]):
			values = data.get_act_all_examples(layer, node)
			deviation += np.std(values)
	return deviation /float(sum(hidden_nodes))
	
def class_accuracy(data, dnf, target_class_index, t_v = True, tr = False, v = False, te = False):
	'''
	Determines the accuracy of the dnf of different dataset splits with regards to the class value
	'''
	result = []
	if t_v:
		n_examples = data.num_train + data.num_vali
		consistent = sum([1 for e in data.get_train_vali_obs() if (target_class_index == e.class_value) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	if tr:
		n_examples = data.num_train
		consistent = sum([1 for e in data.get_train_obs() if (target_class_index == e.class_value) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	if v:
		n_examples = data.num_vali
		consistent = sum([1 for e in data.get_vali_obs() if (target_class_index == e.class_value) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	if te:
		n_examples = data.num_test
		consistent = sum([1 for e in data.get_test_obs() if (target_class_index == e.class_value) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	return result

def prediction_fidelity(data, dnf, target_class_index, t_v = True, tr = False, v = False, te = False):
	'''
	Determines the fidelity of the dnf of different dataset splits with regards to the predictions of the network
	'''
	result = []
	if t_v:
		n_examples = data.num_train + data.num_vali
		consistent = sum([1 for e in data.get_train_vali_obs() if (target_class_index == e.orig_prediction) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	if tr:
		n_examples = data.num_train
		consistent = sum([1 for e in data.get_train_obs() if (target_class_index == e.orig_prediction) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	if v:
		n_examples = data.num_vali
		consistent = sum([1 for e in data.get_vali_obs() if (target_class_index == e.orig_prediction) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	if te:
		n_examples = data.num_test
		consistent = sum([1 for e in data.get_test_obs() if (target_class_index == e.orig_prediction) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	return result
	
def class_precision(data, dnf, target_class_index, t_v = True, tr = False, v = False, te = False):
	'''
	Determines the precision of the dnf of different dataset splits with regards to the class value
	'''
	result = []
	if t_v:
		tp = sum([1 for e in data.get_train_vali_obs() if (target_class_index == e.class_value) and e.fulfills_dnf(dnf)])
		fp = sum([1 for e in data.get_train_vali_obs() if (target_class_index != e.class_value) and e.fulfills_dnf(dnf)])
		if tp+fp > 0:
			result.append(float(tp)/float(tp+fp))
		else:
			result.append(1)
	if tr:
		tp = sum([1 for e in data.get_train_obs() if (target_class_index == e.class_value) and e.fulfills_dnf(dnf)])
		fp = sum([1 for e in data.get_train_obs() if (target_class_index != e.class_value) and e.fulfills_dnf(dnf)])
		if tp+fp > 0:
			result.append(float(tp)/float(tp+fp))
		else:
			result.append(1)
	if v:
		tp = sum([1 for e in data.get_vali_obs() if (target_class_index == e.class_value) and e.fulfills_dnf(dnf)])
		fp = sum([1 for e in data.get_vali_obs() if (target_class_index != e.class_value) and e.fulfills_dnf(dnf)])
		if tp+fp > 0:
			result.append(float(tp)/float(tp+fp))
		else:
			result.append(1)
	if te:
		tp = sum([1 for e in data.get_test_obs() if (target_class_index == e.class_value) and e.fulfills_dnf(dnf)])
		fp = sum([1 for e in data.get_test_obs() if (target_class_index != e.class_value) and e.fulfills_dnf(dnf)])
		if tp+fp > 0:
			result.append(float(tp)/float(tp+fp))
		else:
			result.append(1)
	return result

def class_recall(data, dnf, target_class_index, t_v = True, tr = False, v = False, te = False):
	'''
	Determines the recall of the dnf of different dataset splits with regards to the class value
	'''
	result = []
	if t_v:
		tp = sum([1 for e in data.get_train_vali_obs() if (target_class_index == e.class_value) and e.fulfills_dnf(dnf)])
		fn = sum([1 for e in data.get_train_vali_obs() if (target_class_index == e.class_value) and (not e.fulfills_dnf(dnf))])
		if tp+fn > 0:
			result.append(float(tp)/float(tp+fn))
		else:
			result.append(1)
	if tr:
		tp = sum([1 for e in data.get_train_obs() if (target_class_index == e.class_value) and e.fulfills_dnf(dnf)])
		fn = sum([1 for e in data.get_train_obs() if (target_class_index == e.class_value) and (not e.fulfills_dnf(dnf))])
		if tp+fn > 0:
			result.append(float(tp)/float(tp+fn))
		else:
			result.append(1)
	if v:
		tp = sum([1 for e in data.get_vali_obs() if (target_class_index == e.class_value) and e.fulfills_dnf(dnf)])
		fn = sum([1 for e in data.get_vali_obs() if (target_class_index == e.class_value) and (not e.fulfills_dnf(dnf))])
		if tp+fn > 0:
			result.append(float(tp)/float(tp+fn))
		else:
			result.append(1)
	if te:
		tp = sum([1 for e in data.get_test_obs() if (target_class_index == e.class_value) and e.fulfills_dnf(dnf)])
		fn = sum([1 for e in data.get_test_obs() if (target_class_index == e.class_value) and (not e.fulfills_dnf(dnf))])
		if tp+fn > 0:
			result.append(float(tp)/float(tp+fn))
		else:
			result.append(1)
	return result

def accuracy_of_dnf(data, key, dnf, t_v = True, tr = False, v = False, te = False):
	'''
	Determines the accuracy of a dnf with respect to its class condition
	'''
	result = []
	if t_v:
		n_examples = data.num_train + data.num_vali
		consistent = sum([1 for e in data.get_train_vali_obs() if e.fulfills_cond(key) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	if tr:
		n_examples = data.num_train
		consistent = sum([1 for e in data.get_train_obs() if e.fulfills_cond(key) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	if v:
		n_examples = data.num_vali
		consistent = sum([1 for e in data.get_vali_obs() if e.fulfills_cond(key) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	if te:
		n_examples = data.num_test
		consistent = sum([1 for e in data.get_test_obs() if e.fulfills_cond(key) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	return result

def precision_of_dnf(data, key, dnf, t_v = True, tr = False, v = False, te = False):
	'''
	Determines the precision of a dnf with respect to its class condition
	'''
	result = []
	if t_v:
		tp = sum([1 for e in data.get_train_vali_obs() if e.fulfills_cond(key) and e.fulfills_dnf(dnf)])
		fp = sum([1 for e in data.get_train_vali_obs() if (not e.fulfills_cond(key)) and e.fulfills_dnf(dnf)])
		if tp+fp > 0:
			result.append(float(tp)/float(tp+fp))
		else:
			result.append(1)
	if tr:
		tp = sum([1 for e in data.get_train_obs() if e.fulfills_cond(key) and e.fulfills_dnf(dnf)])
		fp = sum([1 for e in data.get_train_obs() if (not e.fulfills_cond(key)) and e.fulfills_dnf(dnf)])
		if tp+fp > 0:
			result.append(float(tp)/float(tp+fp))
		else:
			result.append(1)
	if v:
		tp = sum([1 for e in data.get_vali_obs() if e.fulfills_cond(key) and e.fulfills_dnf(dnf)])
		fp = sum([1 for e in data.get_vali_obs() if (not e.fulfills_cond(key)) and e.fulfills_dnf(dnf)])
		if tp+fp > 0:
			result.append(float(tp)/float(tp+fp))
		else:
			result.append(1)
	if te:
		tp = sum([1 for e in data.get_test_obs() if e.fulfills_cond(key) and e.fulfills_dnf(dnf)])
		fp = sum([1 for e in data.get_test_obs() if (not e.fulfills_cond(key)) and e.fulfills_dnf(dnf)])
		if tp+fp > 0:
			result.append(float(tp)/float(tp+fp))
		else:
			result.append(1)
	return result
	
def recall_of_dnf(data, key, dnf, t_v = True, tr = False, v = False, te = False):
	'''
	Determines the recall of a dnf with respect to its class condition
	'''
	result = []
	if t_v:
		tp = sum([1 for e in data.get_train_vali_obs() if e.fulfills_cond(key) and e.fulfills_dnf(dnf)])
		fn = sum([1 for e in data.get_train_vali_obs() if e.fulfills_cond(key) and (not e.fulfills_dnf(dnf))])
		if tp+fn > 0:
			result.append(float(tp)/float(tp+fn))
		else:
			result.append(1)
	if tr:
		tp = sum([1 for e in data.get_train_obs() if e.fulfills_cond(key) and e.fulfills_dnf(dnf)])
		fn = sum([1 for e in data.get_train_obs() if e.fulfills_cond(key) and (not e.fulfills_dnf(dnf))])
		if tp+fn > 0:
			result.append(float(tp)/float(tp+fn))
		else:
			result.append(1)
	if v:
		tp = sum([1 for e in data.get_vali_obs() if e.fulfills_cond(key) and e.fulfills_dnf(dnf)])
		fn = sum([1 for e in data.get_vali_obs() if e.fulfills_cond(key) and (not e.fulfills_dnf(dnf))])
		if tp+fn > 0:
			result.append(float(tp)/float(tp+fn))
		else:
			result.append(1)
	if te:
		tp = sum([1 for e in data.get_test_obs() if e.fulfills_cond(key) and e.fulfills_dnf(dnf)])
		fn = sum([1 for e in data.get_test_obs() if e.fulfills_cond(key) and (not e.fulfills_dnf(dnf))])
		if tp+fn > 0:
			result.append(float(tp)/float(tp+fn))
		else:
			result.append(1)
	return result

def example_indexes(data, key, dnf, t_v = True, tr = False, v = False, te = False):
	'''
	Returns a tupel of the examples from each set that are inconsistent
	'''
	result = []
	if t_v:
		result.append([e.idx for e in data.get_train_vali_obs() if e.fulfills_cond(key) != e.fulfills_dnf(dnf)])
	if tr:
		result.append([e.idx for e in data.get_train_obs() if e.fulfills_cond(key) != e.fulfills_dnf(dnf)])
	if v:
		result.append([e.idx for e in data.get_vali_obs() if e.fulfills_cond(key) != e.fulfills_dnf(dnf)])
	if te:
		result.append([e.idx for e in data.get_test_obs() if e.fulfills_cond(key) != e.fulfills_dnf(dnf)])
	return result


def split_points(dnf):
	sp = set([])
	if isinstance(dnf, list):
		for rule in dnf:
			if isinstance(rule, list):
				for c in rule:
					if isinstance(c, tuple):
						sp.add((c[0], c[1], c[2]))
	#return set([(l, n, t) for rule in dnf for (l, n, t, b) in rule])
	return sp

def number_conditions(dnf):
	num_conditions = 0
	if isinstance(dnf, list):
		for rule in dnf:
			if isinstance(rule, list):
				for c in rule:
					if isinstance(c, tuple):
						num_conditions += 1
	return num_conditions

def number_rules(dnf):
	if isinstance(dnf, list):
		return len(dnf)
	else:
		return 1

def num_distinct_split_points(dnf):
	return len(split_points(dnf))

def number_entries(BNN):
	return len(BNN)-1 # minus the outputs

def BNN_number_conditions(BNN):
	return sum(number_conditions(dnf) for dnf in BNN.values())

def BNN_number_rules(BNN):
	return sum(number_rules(dnf) for dnf in BNN.values())

def BNN_num_distinct_split_points(BNN):
	distinct = set([])
	for dnf in BNN.values():
		distinct.update(split_points(dnf))
	distinct = [(l, n, t) for (l, n, t) in distinct if l>0]
	return len(distinct)-1

def BNN_avg_thresholds_used_neurons(BNN):
	distinct = set([])
	for dnf in BNN.values():
		distinct.update(split_points(dnf))
	distinct = [(l, n, t) for (l, n, t) in distinct if l>0]
	used_neurons = set((l, n) for (l, n, t) in distinct)
	l_u_n = len(used_neurons) -1 # minus the output threshold
	if l_u_n==0:
		l_u_n = 1
	thresholds = len(distinct) -1
	return float(thresholds)/l_u_n

def per_layer_info(data, BNN, layers):
	keys = BNN.keys()
	num_conds = [None] * layers
	train_fidelity = [None] * layers
	test_fidelity = [None] * layers
	for layer in range(layers):
		conds = [(l, n, t, b) for (l, n, t, b) in keys if l == layer+1]
		num_conds[layer] = sum(1 for c in conds)
		if num_conds[layer] > 0:
			train_fidelity[layer] = float(sum(accuracy_of_dnf(data, c, BNN[c], t_v = False, tr = True, v = False, te = False)[0] for c in conds))/num_conds[layer]
			test_fidelity[layer] = float(sum(accuracy_of_dnf(data, c, BNN[c], t_v = False, tr = False, v = False, te = True)[0] for c in conds))/num_conds[layer]
		else:
			train_fidelity[layer] = 0
			test_fidelity[layer] = 0
	return num_conds, train_fidelity, test_fidelity

