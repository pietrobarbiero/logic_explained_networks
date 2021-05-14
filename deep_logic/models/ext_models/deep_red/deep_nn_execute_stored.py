# Converts a manually inputted network to a Tensorflow model. I used this 
# to sample the XOR network Jan Zilke used in the DeepRED paper.

import numpy as np
import matplotlib
import tensorflow as tf
from numpy import genfromtxt
import time

def init_weights(shape, i):
	weights_name = 'W'
	weights_name += str(i)
	return tf.Variable(tf.random_uniform(shape, -1, 1), name = weights_name)

def init_bias(size, i):
	bias_name = 'B'
	bias_name += str(i)
	return tf.Variable(tf.random_uniform([size], -1, 1), name = bias_name)

def accuracy(x, one_hot_y, hypothesis):
	number_examples = len(x)
	correct = 0
	for e in range(number_examples):
		prediction = list(hypothesis[e])
		if one_hot_y[e].index(max(one_hot_y[e])) == prediction.index(max(prediction)):
			correct += 1
	return float(correct)/float(number_examples)

def construct_objects(data, model_name, hidden_nodes, weights, bias, softmax=True, store_adam_vars=True):
	x_train, y_train = data.get_train_x_y()
	x_test, y_test = data.get_test_x_y()

	input_size = len(x_train[0])
	output_size = len(y_train[0])
	
	layers = len(hidden_nodes)+1
	
	X_train = tf.placeholder(tf.float32, shape=[len(x_train),input_size])
	Y_train = tf.placeholder(tf.float32, shape=[len(x_train),output_size])
	X_test = tf.placeholder(tf.float32, shape=[len(x_test),input_size])
	Y_test = tf.placeholder(tf.float32, shape=[len(x_test),output_size])
	
	# Initial weights and bias are set
	W = [None]*layers
	B = [None]*layers

	for i in range(layers):
		weights_name = 'W' + str(i)
		bias_name = 'B' + str(i)
		print('i', i)
		W[i] = tf.Variable(weights[i], dtype=tf.float32, name = weights_name)
		B[i] = tf.Variable(bias[i], dtype=tf.float32, name = bias_name)

	# Lists that store the activation values
	A_train = [None]*layers
	A_test = [None]*layers

	A_train[0] = tf.sigmoid(tf.matmul(X_train, W[0]) + B[0])
	A_test[0] = tf.sigmoid(tf.matmul(X_test, W[0]) + B[0])
	
	for i in range(1,layers-1):
		A_train[i] = tf.sigmoid(tf.matmul(A_train[i-1], W[i]) + B[i])
		A_test[i] = tf.sigmoid(tf.matmul(A_test[i-1], W[i]) + B[i])
		
	if softmax:
		# Softmax layer
		logits = tf.matmul(A_train[layers-2], W[layers-1]) + B[layers-1]
		A_train[layers-1] = tf.nn.softmax(logits)
		A_test[layers-1] = tf.nn.softmax(tf.matmul(A_test[layers-2], W[layers-1]) + B[layers-1])
	else:
		logits = tf.matmul(A_train[layers-2], W[layers-1]) + B[layers-1]
		A_train[layers-1] = logits
		A_test[layers-1] = tf.sigmoid(tf.matmul(A_test[layers-2], W[layers-1]) + B[layers-1])
	Hypothesis_train = A_train[layers-1]
	Hypothesis_test = A_test[layers-1]
	
	if softmax:
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, Y_train))
	else:
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, Y_train))
		
	train_step = tf.train.AdamOptimizer().minimize(loss)
	
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)

	t_start = time.clock()

	h_train = sess.run(Hypothesis_train, feed_dict={X_train: x_train})
	print('TRAIN ACCURACY', accuracy(x_train, y_train, h_train))
	h_test = sess.run(Hypothesis_test, feed_dict={X_test: x_test})
	print('TEST ACCURACY', accuracy(x_test, y_test, h_test))

	if store_adam_vars:
		sess.run(train_step, feed_dict={X_train: x_train, Y_train: y_train})

	# Save the variables to disk
	saver = tf.train.Saver()
	save_path = saver.save(sess, model_name)
	print("Model saved in file: %s" % save_path)

	t_end = time.clock()
	passed_time = 'Passed time: ' + str(t_end - t_start)
	print(passed_time)
	sess.close()


def xor_8_zilke(data, network_model_name, hidden_nodes):
	W0 = np.asmatrix(np.array([
	[7.0, -7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
	[7.0, -7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
	[0.0, 0.0, 7.0, -7.0, 0.0, 0.0, 0.0, 0.0],
	[0.0, 0.0, 7.0, -7.0, 0.0, 0.0, 0.0, 0.0],
	[0.0, 0.0, 0.0, 0.0, 7.0, -7.0, 0.0, 0.0],
	[0.0, 0.0, 0.0, 0.0, 7.0, -7.0, 0.0, 0.0],
	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, -7.0],
	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, -7.0]]))

	W1 = np.asmatrix(np.array([
	[-7.0, 0.0, 0.0, 0.0],
	[-7.0, 0.0, 0.0, 0.0],
	[0.0, -7.0, 0.0, 0.0],
	[0.0, -7.0, 0.0, 0.0],
	[0.0, 0.0, -7.0, 0.0],
	[0.0, 0.0, -7.0, 0.0],
	[0.0, 0.0, 0.0, -7.0],
	[0.0, 0.0, 0.0, -7.0]]))

	W2 = np.asmatrix(np.array([
	[7.0, -7.0, 0.0, 0.0],
	[7.0, -7.0, 0.0, 0.0],
	[0.0, 0.0, 7.0, -7.0],
	[0.0, 0.0, 7.0, -7.0]]))

	W3 = np.asmatrix(np.array([
	[-7.0, 0.0],
	[-7.0, 0.0],
	[0.0, -7.0],
	[0.0, -7.0]]))

	W4 = np.asmatrix(np.array([
	[7.0, -7.0],
	[7.0, -7.0]]))
	
	W5 = np.asmatrix(np.array([
	[7.0, -7.0],
	[7.0, -7.0]]))

	B0 = np.array([-10.0, 3.0, -10.0, 3.0, -10.0, 3.0, -10.0, 3.0])
	B1 = np.array([3.0, 3.0, 3.0, 3.0])
	B2 = np.array([-10.0, 3.0, -10.0, 3.0])
	B3 = np.array([3.0, 3.0])
	B4 = np.array([-10.0, 3.0])
	B5 = np.array([-3.0, 3.0])

	weights = [W0, W1, W2, W3, W4, W5]
	bias = [B0, B1, B2, B3, B4, B5]
	
	return construct_objects(data, network_model_name, hidden_nodes, weights, bias, False, False)


