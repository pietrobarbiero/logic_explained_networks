import numpy as np
import matplotlib
import itertools
import tensorflow.compat.v1 as tf
import time
from operator import concat
import copy
import random

tf.disable_v2_behavior()
tf.disable_eager_execution()

def init_weights(shape, i):
	weights_name = 'W'
	weights_name += str(i)
	return tf.Variable(tf.random_uniform(shape, -1, 1), name = weights_name)

def init_bias(size, i):
	bias_name = 'B'
	bias_name += str(i)
	return tf.Variable(tf.random_uniform([size], -1, 1), name = bias_name)

def one_hot_code(hypothesis):
	number_examples = len(hypothesis)
	neurons = len(hypothesis[0])
	one_hot_coded_h = [None] * number_examples
	for e in range(number_examples):
		one_hot_coded_h[e] = [0] * neurons
		h = list(hypothesis[e])
		higher = h.index(max(h))
		one_hot_coded_h[e][higher] = 1
	return one_hot_coded_h
		
def accuracy(x, one_hot_y, hypothesis):
	number_examples = len(x)
	correct = 0
	for e in range(number_examples):
		prediction = list(hypothesis[e])
		if one_hot_y[e].index(max(one_hot_y[e])) == prediction.index(max(prediction)):
			correct += 1
	return float(correct)/float(number_examples)

def classification_example_indexes(x, one_hot_y, hypothesis):
	'''
	Returns a tuple of two sets, where the first element is the indexes
	that were correctly classified and the second, those that were not.
	These are not the indexes in the actual dataset but rather refer to 
	the position of the observation on the train set
	'''
	correct = set([])
	misses = set([])
	for e in range(len(x)):
		prediction = list(hypothesis[e])
		if one_hot_y[e].index(max(one_hot_y[e])) == prediction.index(max(prediction)):
			correct.add(e)
		else:
			misses.add(e)
	return correct, misses

def smallest_matrix_index(w, indexes):
	'''
	Returns the index of the smallest entry in the matrix that is not
	already in 'indexes'
	'''
	remaining_indexes = [i for i in itertools.product(range(w.shape[0]), 
					range(w.shape[1])) if i not in indexes]
	return min(remaining_indexes, key= lambda i: abs(w.item(i)))

def remake_mask(shape, indexes):
	'''
	It creates a matrix of some shape where all entries are one except
	those whose entries specified in 'indexes', which are zero.
	'''
	w = np.ones(shape)
	for i in indexes:
		w.itemset(i, 0)
	return w

def indexes_of_neuron(shape, shallow_neuron_index):
	'''
	Returns all indexes that mark the outgoing connections from a neuron
	'''
	return [(shallow_neuron_index, i) for i in range(shape[1])]


def retrain_network(data, model_name, new_model_name, hidden_nodes, iterations, batch_size=0):
	'''
	Polarizes the activations of a network using the hyperbolic tangent activation function and with a softmax layer at the end.
	
	param data: a DataSet instance
	param model_name: name with which model will be stored
	param hidden_nodes: number of nodes on each hidden layer, as [[3], [4], [4]] for a network with three hidden layers
	param iterations: iterations for training
	param batch_size: size of a training batch, '0' means no batch training
	'''

	BETA = tf.placeholder(tf.float32)
	RHO = tf.constant(0.99)
	
	def logfunc(x, x2):
		return tf.multiply( x, tf.log(tf.divide(x,x2)))
	
	def KL_Div(rho, rho_hat):
		invrho = tf.subtract(tf.constant(1.), rho)
		invrhohat = tf.subtract(tf.constant(1.), rho_hat)
		logrho = tf.add(logfunc(rho,rho_hat), logfunc(invrho, invrhohat))
		return logrho
	
	x, y = data.get_train_x_y()
	if data.num_test == 0:
		x_test, y_test = data.get_train_x_y()
	else:
		x_test, y_test = data.get_test_x_y()

	input_size = len(x[0])
	output_size = len(y[0])
	
	layers = len(hidden_nodes)+1
	
	if batch_size == 0:
		batch_size = len(x)
	
	X_train = tf.placeholder(tf.float32, shape=[batch_size,input_size])
	Y_train = tf.placeholder(tf.float32, shape=[batch_size,output_size])
	
	X_test = tf.placeholder(tf.float32, shape=[len(x_test),input_size])
	Y_test = tf.placeholder(tf.float32, shape=[len(x_test),output_size])
	
	keep_prob = tf.placeholder(tf.float32)
	
	# Initial weights and bias are set
	W = [None]*layers
	masks = [None]*layers
	B = [None]*layers
	W[0] = init_weights([input_size, hidden_nodes[0]], 0)
	masks[0] = tf.ones([input_size, hidden_nodes[0]], dtype=tf.float32, name=None)
	W[layers-1] = init_weights([hidden_nodes[layers-2], output_size], layers-1)
	masks[layers-1] = tf.ones([hidden_nodes[layers-2], output_size], dtype=tf.float32, name=None)
	B[0] = init_bias(hidden_nodes[0], 0)
	B[layers-1] = init_bias(output_size, layers-1)
	for i in range(layers-2):
		W[i+1] = init_weights([hidden_nodes[i], hidden_nodes[i+1]], i+1)
		B[i+1] = init_bias(hidden_nodes[i+1], i+1)
		masks[i+1] = tf.ones([hidden_nodes[i], hidden_nodes[i+1]], dtype=tf.float32, name=None)
	
	# Lists that store the activation values
	A_train = [None]*layers
	A_test = [None]*layers
	
	rho_hat = [None]*(layers-1) # Has the means for all layers
	divergance = [None]*(layers-1)
	
	A_train[0] = tf.tanh(tf.matmul(X_train, masks[0]*W[0]) + B[0])
	A_test[0] = tf.tanh(tf.matmul(X_test, masks[0]*W[0]) + B[0])

	rho_hat[0] = tf.divide(tf.reduce_sum(tf.abs(A_train[0]),0),tf.constant(float(batch_size)))
	divergance[0] = tf.reduce_sum(KL_Div(RHO, rho_hat[0]))

	for i in range(1,layers-1):
		A_train[i] = tf.tanh(tf.matmul(tf.nn.dropout(A_train[i-1], keep_prob), masks[i]*W[i]) + B[i])
		A_test[i] = tf.tanh(tf.matmul(A_test[i-1], masks[i]*W[i]) + B[i])
		rho_hat[i] = tf.divide(tf.reduce_sum(tf.abs(A_train[i]),0),tf.constant(float(batch_size)))
		divergance[i] = tf.reduce_sum(KL_Div(RHO, rho_hat[i]))
			
	# Softmax layer
	logits = tf.matmul(A_train[layers-2], masks[layers-1]*W[layers-1]) + B[layers-1]
	A_train[layers-1] = tf.nn.softmax(logits)
	A_test[layers-1] = tf.nn.softmax(tf.matmul(A_test[layers-2], masks[layers-1]*W[layers-1]) + B[layers-1])
	Hypothesis_train = A_train[layers-1]
	Hypothesis_test = A_test[layers-1]
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits, Y_train)) #ADDED v2 in TF 2.0
	cost_sparse = tf.reduce_sum(divergance)
	cost = tf.add(loss , tf.multiply(BETA, cost_sparse))
	train_step = tf.train.AdamOptimizer().minimize(cost)

	# Add an op to initialize the variables
	init = tf.initialize_all_variables()

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	sess = tf.Session()

	# Restore trained network
	sess.run(init)
	saver.restore(sess, 'models/'+model_name+'.ckpt')
	print("Model " + model_name + " restored")

	weights = [None]*layers
	bias = [None]*layers
	
	x_train, y_train = x, y
	h_test = sess.run(Hypothesis_test, feed_dict={X_test: x_test, keep_prob: 1})
	initial_accuracy = accuracy(x_test, y_test, h_test) - 0.01
	new_accuracy = initial_accuracy
	cost_pol = 1
	bet = 0.0
	
	i = 0
	while i == 0:
	#while new_accuracy >= initial_accuracy and cost_pol>0.1:
		for j in range(layers):
			weights[j] = sess.run(W[j])
			bias[j] = sess.run(B[j])
		bet += 0.1
		print('BETA', bet)
		for i in range(iterations):
			if batch_size > 0:
				batch_indexes = random.sample(range(len(x_train)), batch_size)
				x_train = [e for (j, e) in enumerate(x) if j in batch_indexes]
				y_train = [e for (j, e) in enumerate(y) if j in batch_indexes]
			sess.run(train_step, feed_dict={X_train: x_train, Y_train: y_train, keep_prob: 1, BETA: bet})
			if i % (iterations/10) == 0:
				print('EPOCH ', i)
				print('Cost sparse', sess.run(cost_sparse, feed_dict={X_train: x_train, Y_train: y_train, keep_prob: 1, BETA: bet}))
				print('Loss ', sess.run(loss, feed_dict={X_train: x_train, Y_train: y_train, keep_prob: 1, BETA: bet}))
				cost_pol = sess.run(cost_sparse, feed_dict={X_train: x_train, Y_train: y_train, keep_prob: 1, BETA: bet})
				print('COST ', sess.run(cost, feed_dict={X_train: x_train, Y_train: y_train, keep_prob: 1, BETA: bet}))
				h_test = sess.run(Hypothesis_test, feed_dict={X_test: x_test, keep_prob: 1})
				new_accuracy = accuracy(x_test, y_test, h_test)
				print('TRAIN ACCURACY', new_accuracy)
		i = 1

	# Save the variables to disk
	save_path = saver.save(sess, 'models/'+new_model_name+'.ckpt')
	print("Model saved in file: %s" % save_path)
	h_train = sess.run(Hypothesis_train, feed_dict={X_train: x_train, keep_prob: 1, BETA: bet})
	new_accuracy = accuracy(x_train, y_train, h_train)

	tf.reset_default_graph()
	h_test = sess.run(Hypothesis_test, feed_dict={X_test: x_test, keep_prob: 1})
	tf.reset_default_graph()
	return bet

def keep_training_wsp_polarize(data, bet, model_name, new_model_name, hidden_nodes, iterations, accuracy_decrease = 0.01, indexes = [], to_prune=[], together = 1, batch_size = 0, max_non_improvements=2, max_runs=5):
	'''
	Performs Weight Sparseness Pruning while continuing to keep the activations polarized
	of a network using the hyperbolic tangent activation function and with a softmax layer at the end.
	
	param data: a DataSet instance
	param bet: last used weighting factor of the penalty term, returned by 'retrain_network'
	param model_name: name with which model will be stored
	param hidden_nodes: number of nodes on each hidden layer, as [[3], [4], [4]] for a network with three hidden layers
	param iterations: iterations for training
	param accuracy_decrease: allowed accuracy decrease for a connection to remain pruned
	param together: number of connections that are pruned together
	param batch_size: size of a training batch, '0' means no batch training
	param max_non_improvements: after pruning some index, number of times a retraining iteration set is performed without 
	increase of accuracy before the connection is deemed not prunable
	param max_runs: after pruning some index, number of times a retraining iteration set is performed before the connection 
	is deemed not prunable
	'''

	BETA = tf.constant(bet)
	RHO = tf.constant(0.99)
	
	def logfunc(x, x2):
		return tf.mul( x, tf.log(tf.div(x,x2)))
	
	def KL_Div(rho, rho_hat):
		invrho = tf.sub(tf.constant(1.), rho)
		invrhohat = tf.sub(tf.constant(1.), rho_hat)
		logrho = tf.add(logfunc(rho,rho_hat), logfunc(invrho, invrhohat))
		return logrho
	
	def remake_mask(shape, indexes):
		'''
		It creates a matrix of some shape where all entries are one except
		those whose entries specified in 'indexes', which are zero.
		
		param shape: shape of the mask
		shape type: (int, int)
		param indexes: for each layer, the indexes are stores which are currently pruned
		indexes type: list of lists, where each list element stores (int, int) tuples
		'''
		w = np.ones(shape)
		for i in indexes:
			w.itemset(i, 0)
		return w

	def initialize_possible_indexes(indexes, to_prune, shapes):
		'''
		It creates a matrix of some shape where all entries are one except
		those whose entries specified in 'indexes', which are zero.
		
		param shapes: list of shapes
		shape type: list of (int, int) tuples
		param to_prune: list of matrix indexes
		to_prune type: list of ints
		param indexes: for each layer, the indexes are stores which are currently pruned
		indexes type: list of lists, where each list element stores (int, int) tuples
		return: a list of (m, r, c) elements, where m is the matrix identifier and (r, c) 
		the indexes to prune
		rtype: list of (int, int, int) tuples
		'''
		return reduce(concat, [[(h, i, j) for (i, j) in 
		list(itertools.product(range(shapes[h][0]), range(shapes[h][1]))) 
		if (i, j) not in indexes[h]] for h in to_prune])

	def initialize_index_counts(indexes, to_prune, shapes):
		'''
		It initializes the row and matrix counts.
		
		param shapes: list of shapes
		shape type: list of (int, int) tuples
		param to_prune: list of matrix indexes
		to_prune type: list of ints
		param indexes: for each layer, the indexes are stored which are currently pruned
		indexes type: list of lists, where each list element stores (int, int) tuples
		return: a dictionary of (m, r) for rows and one of (m, c) for columns
		rtype: two dictionaries, each of (int, int) : int
		'''
		rows = {}
		cols = {}
		for m in to_prune:
			poss_rows = range(shapes[m][0])
			for pr in poss_rows:
				rows[(m, pr)] = sum(1 for (i, j) in indexes[m] if i==pr)
			poss_cols = range(shapes[m][1])
			for pc in poss_cols:
				cols[(m, pc)] = sum(1 for (i, j) in indexes[m] if j==pc)
		return rows, cols

	x, y = data.get_train_x_y()
	if data.num_vali == 0:
		x_vali, y_vali = data.get_train_x_y()
	else:
		x_vali, y_vali = data.get_vali_x_y()
		
	input_size = len(x[0])
	output_size = len(y[0])
		
	layers = len(hidden_nodes)+1

	if batch_size == 0:
		batch_size = len(x)
	
	X_train = tf.placeholder(tf.float32, shape=[batch_size,input_size])
	Y_train = tf.placeholder(tf.float32, shape=[batch_size,output_size])
	X_vali = tf.placeholder(tf.float32, shape=[len(x_vali),input_size])
	Y_vali = tf.placeholder(tf.float32, shape=[len(x_vali),output_size])

	W = [None]*layers
	masks = [None]*layers
	B = [None]*layers
	
	rho_hat = [None]*(layers-1) # Has the means for all layers
	divergance = [None]*(layers-1)
	
	W[0] = init_weights([input_size, hidden_nodes[0]], 0)
	masks[0] = tf.placeholder(tf.float32, shape=[input_size, hidden_nodes[0]])
	W[layers-1] = init_weights([hidden_nodes[layers-2], output_size], layers-1)
	masks[layers-1] = tf.placeholder(tf.float32, shape=[hidden_nodes[layers-2], output_size])
	B[0] = init_bias(hidden_nodes[0], 0)
	B[layers-1] = init_bias(output_size, layers-1)
	
	for i in range(layers-2):
		W[i+1] = init_weights([hidden_nodes[i], hidden_nodes[i+1]], i+1)
		B[i+1] = init_bias(hidden_nodes[i+1], i+1)
		masks[i+1] = tf.placeholder(tf.float32, shape=[hidden_nodes[i], hidden_nodes[i+1]])
	
	# Lists that store the activation values
	A_train = [None]*layers
	A_vali = [None]*layers
	
	A_train[0] = tf.tanh(tf.matmul(X_train, masks[0]*W[0]) + B[0])
	A_vali[0] = tf.tanh(tf.matmul(X_vali, masks[0]*W[0]) + B[0])
	rho_hat[0] = tf.div(tf.reduce_sum(tf.abs(A_train[0]),0),tf.constant(float(batch_size)))
	divergance[0] = tf.reduce_sum(KL_Div(RHO, rho_hat[0]))
	
	for i in range(1,layers-1):
		A_train[i] = tf.tanh(tf.matmul(A_train[i-1], masks[i]*W[i]) + B[i])
		A_vali[i] = tf.tanh(tf.matmul(A_vali[i-1], masks[i]*W[i]) + B[i])
		rho_hat[i] = tf.div(tf.reduce_sum(tf.abs(A_train[i]),0),tf.constant(float(batch_size)))
		divergance[i] = tf.reduce_sum(KL_Div(RHO, rho_hat[i]))
		
	# Softmax layer
	logits = tf.matmul(A_train[layers-2], masks[layers-1]*W[layers-1]) + B[layers-1]
	A_train[layers-1] = tf.nn.softmax(logits)
	A_vali[layers-1] = tf.nn.softmax(tf.matmul(A_vali[layers-2], masks[layers-1]*W[layers-1]) + B[layers-1])
	Hypothesis_train = A_train[layers-1]
	Hypothesis_vali = A_vali[layers-1]


	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, Y_train))
	cost_sparse = tf.reduce_sum(divergance)
	cost = tf.add(loss , tf.mul(BETA, cost_sparse))
	train_step = tf.train.AdamOptimizer().minimize(cost)

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	sess = tf.Session()
	writer = tf.train.SummaryWriter("./logs/nn_logs", sess.graph)

	# Restore trained network
	#saver.restore(sess, model_name)	
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	saver.restore(sess, 'models/'+model_name+'.ckpt')
	print("Model " + model_name + " restored")
	t_start = time.clock()

	# If no indexes are specified, start by not pruning any indexes
	if not indexes:
		indexes = [None] * layers
		for i in range(layers):
			indexes[i] = []
	
	# If it is not specified what layers to prune, prune all layers (except for softmax if present)
	to_prune = range(layers-1)
	
	# 'shapes' stores the shape of each matrik, 'masks' the mask each matrix takes
	shapes = [None] * layers
	current_masks = [None] * layers
	for i in range(layers):
		shapes[i] = sess.run(W[i]).shape
		current_masks[i] = remake_mask(shapes[i], indexes[i])
	
	# Possible indexes that can be pruned, entries of the type (m, r, c) 
	left_indexes = initialize_possible_indexes(indexes, to_prune, shapes)
	pos_ind = copy.deepcopy(left_indexes)
	
	# Count of indexes for each matrix
	row_count, col_count = initialize_index_counts(indexes, to_prune, shapes)
	
	# Dictionary of masks, x and y that is passed to the placeholders
	placeholder_dict = {i: d for i, d in zip(masks, current_masks)}
	placeholder_dict.update({X_vali: x_vali})
	
	# Initial validation accuracy
	h_vali = sess.run(Hypothesis_vali, feed_dict=placeholder_dict)
	initial_vali_acc = accuracy(x_vali, y_vali, h_vali)
	target_accuracy = initial_vali_acc-accuracy_decrease
	print('Target accuracy: ', target_accuracy)
	new_accuracy = target_accuracy
	
	# Initial parameters are stored
	weights = [None]*layers
	bias = [None]*layers
	for j in range(layers):
		weights[j] = sess.run(W[j])
		bias[j] = sess.run(B[j])

	while together > 0:
		pos_ind = left_indexes
		left_indexes = []
		# While there are still possible indexes to prune
		while pos_ind:
			# Sort possible indexes in order of increasing pruned connections
			pos_ind.sort(key=lambda m, i, j:row_count[(m, i)] + col_count[(m, j)])
			#sorted_poss_ind = sorted(pos_ind, key=lambda(m, i, j):row_count[(m, i)] + col_count[(m, j)])
			non_found = True
			for i in range(len(pos_ind)):
				selected_indexes = []
				for index in range(together):
					selected_indexes.append(pos_ind.pop(0))
					#(m, r, c) = pos_ind.pop(0)
				
				print('Pruned indexies: '+str(sum(len(l) for l in indexes)))
				print('Remaining indexies: '+str(len(pos_ind)))
				print('left for next: '+str(len(left_indexes)))
				for (m, r, c) in selected_indexes:
					indexes[m].append((r, c))
				for mat in set(m for (m, r, c) in selected_indexes):	
					current_masks[mat] = remake_mask(shapes[mat], indexes[mat])
				
				for j in range(layers):
					assign_op = W[j].assign(W[j]*tf.constant(current_masks[j], dtype=tf.float32))
					sess.run(assign_op)
				placeholder_dict.update({i: d for i, d in zip(masks, current_masks)})
				h_vali = sess.run(Hypothesis_vali, feed_dict=placeholder_dict)
				last_accuracy = accuracy(x_vali, y_vali, h_vali)
				print('First accuracy: ', last_accuracy)
				non_improvements = 0
				runs = 0
				while non_improvements < max_non_improvements and runs < max_runs:
					for i in range(iterations):
						x_train, y_train = x, y
						if batch_size > 0:
							batch_indexes = random.sample(xrange(len(x_train)), batch_size)
							x_train = [e for (j, e) in enumerate(x) if j in batch_indexes]
							y_train = [e for (j, e) in enumerate(y) if j in batch_indexes]
						placeholder_dict.update({X_train: x_train, Y_train: y_train})						
						sess.run(train_step, feed_dict=placeholder_dict)
						#for j in range(layers):
						#	assign_op = W[j].assign(W[j]*tf.constant(current_masks[j], dtype=tf.float32))
						#	sess.run(assign_op)	
						h_vali = sess.run(Hypothesis_vali, feed_dict=placeholder_dict)
						new_accuracy = accuracy(x_vali, y_vali, h_vali)
						if i % (iterations/2) == 0:
							print('Accuracy: ', new_accuracy)
						if new_accuracy >= target_accuracy:
							print('New index added')
							break
					runs += 1
					if new_accuracy >= target_accuracy:
						non_improvements = max_non_improvements
						print('New indexes mainteined')
						for (m, r, c) in selected_indexes:
							row_count[(m, r)] += 1
							col_count[(m, c)] += 1
						for j in range(layers):
							weights[j] = sess.run(W[j])
							bias[j] = sess.run(B[j])
						non_found = False
						break
					elif new_accuracy <= last_accuracy:
						non_improvements += 1
						print('non_improvements', non_improvements)
						last_accuracy = max(new_accuracy, last_accuracy)
					else:
						non_improvements = 0
						print('non_improvements', non_improvements)
						last_accuracy = max(new_accuracy, last_accuracy)
				if non_found:
					print('Recuperating old weights')
					left_indexes.extend(selected_indexes)
					for (m, r, c) in selected_indexes:
						indexes[m].remove((r,c))
					
					for mat in set(m for (m, r, c) in selected_indexes):	
						current_masks[mat] = remake_mask(shapes[mat], indexes[mat])
					placeholder_dict.update({i: d for i, d in zip(masks, current_masks)})
				
					for j in range(layers):
						assign_op = W[j].assign(tf.constant(weights[j], dtype=tf.float32)*tf.constant(current_masks[j], dtype=tf.float32))
						sess.run(assign_op)	
						assign_op = B[j].assign(tf.constant(bias[j], dtype=tf.float32))
						sess.run(assign_op)
					h_vali = sess.run(Hypothesis_vali, feed_dict=placeholder_dict)
					acc = accuracy(x_vali, y_vali, h_vali)
					print('Accuracy: ', acc)
				else:
					print('Index found, break from for')
					break
			if non_found:
				print('Breaking from while, no index can be pruned')
				break
		together = int(float(together)/2)
		
	# Save the variables to disk
	save_path = saver.save(sess, 'models/'+new_model_name+'.ckpt')
	print("Model saved in file: %s" % save_path)
	h_vali = sess.run(Hypothesis_vali, feed_dict=placeholder_dict)
	acc = accuracy(x_vali, y_vali, h_vali)
	print('Final accuracy: ', acc)
	print('MASKS')
	print(current_masks)
	print('WEIGHTS')
	for j in range(layers):
		print(sess.run(W[j]))
		
	tf.reset_default_graph()
	return bet
