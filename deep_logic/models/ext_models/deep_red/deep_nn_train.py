# Trains or retrains network models in different manners, using TensorFlow

import numpy as np
import matplotlib
import itertools
import tensorflow.compat.v1 as tf
import time
from operator import concat
import copy
import random

import functools

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.disable_v2_behavior()
tf.disable_eager_execution()


def init_weights(shape, i):
    weights_name = 'W'
    weights_name += str(i)
    return tf.Variable(tf.random_uniform(shape, -1, 1), name=weights_name)


def init_bias(size, i):
    bias_name = 'B'
    bias_name += str(i)
    return tf.Variable(tf.random_uniform([size], -1, 1), name=bias_name)


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
    return float(correct) / float(number_examples)


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
    return min(remaining_indexes, key=lambda i: abs(w.item(i)))


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


def train_network(data, model_name, hidden_nodes, iterations, function='tanh', softmax=True, batch_size=0):
    '''
	Trains a neural network
	
	param data: a DataSet instance
	param model_name: name with which model will be stored
	param hidden_nodes: number of nodes on each hidden layer, as [[3], [4], [4]] for a network with three hidden layers
	param iterations: iterations for training
	param function: activation function, 'tanh' or 'sigmoid', but can be adapted for other functions
	param softmax: softmax layer at the end?
	param batch_size: size of a training batch, '0' means no batch training
	'''

    x, y = data.get_train_x_y()
    if data.num_test == 0:
        x_test, y_test = data.get_train_x_y()
    else:
        x_test, y_test = data.get_test_x_y()

    input_size = len(x[0])
    output_size = len(y[0])

    layers = len(hidden_nodes) + 1

    if batch_size == 0:
        batch_size = len(x)

    X_train = tf.placeholder(tf.float32, shape=[batch_size, input_size])
    Y_train = tf.placeholder(tf.float32, shape=[batch_size, output_size])

    X_test = tf.placeholder(tf.float32, shape=[len(x_test), input_size])
    Y_test = tf.placeholder(tf.float32, shape=[len(x_test), output_size])

    rate = tf.placeholder(tf.float32)
    # rate = 1 - keep_prob

    # Initial weights and bias are set
    W = [None] * layers
    masks = [None] * layers
    B = [None] * layers
    W[0] = init_weights([input_size, hidden_nodes[0]], 0)
    masks[0] = tf.ones([input_size, hidden_nodes[0]], dtype=tf.float32, name=None)
    W[layers - 1] = init_weights([hidden_nodes[layers - 2], output_size], layers - 1)
    masks[layers - 1] = tf.ones([hidden_nodes[layers - 2], output_size], dtype=tf.float32, name=None)
    B[0] = init_bias(hidden_nodes[0], 0)
    B[layers - 1] = init_bias(output_size, layers - 1)

    for i in range(layers - 2):
        W[i + 1] = init_weights([hidden_nodes[i], hidden_nodes[i + 1]], i + 1)
        B[i + 1] = init_bias(hidden_nodes[i + 1], i + 1)
        masks[i + 1] = tf.ones([hidden_nodes[i], hidden_nodes[i + 1]], dtype=tf.float32, name=None)

    # List that store the activation values
    A_train = [None] * layers
    A_test = [None] * layers

    if function == 'tanh':
        A_train[0] = tf.tanh(tf.matmul(X_train, masks[0] * W[0]) + B[0])
        A_test[0] = tf.tanh(tf.matmul(X_test, masks[0] * W[0]) + B[0])
    else:
        A_train[0] = tf.sigmoid(tf.matmul(X_train, masks[0] * W[0]) + B[0])
        A_test[0] = tf.sigmoid(tf.matmul(X_test, masks[0] * W[0]) + B[0])

    for i in range(1, layers - 1):
        if function == 'tanh':
            # A_train[i] = tf.tanh(tf.matmul(A_train[i-1], masks[i]*W[i]) + B[i])
            A_train[i] = tf.tanh(
                tf.matmul(tf.nn.dropout(A_train[i - 1], rate), masks[i] * W[i]) + B[i])  # ADDED rate for TF v2
            A_test[i] = tf.tanh(tf.matmul(A_test[i - 1], masks[i] * W[i]) + B[i])
        else:
            A_train[i] = tf.sigmoid(tf.matmul(A_train[i - 1], masks[i] * W[i]) + B[i])
            A_test[i] = tf.sigmoid(tf.matmul(A_test[i - 1], masks[i] * W[i]) + B[i])

    if softmax:
        # Softmax layer
        logits = tf.matmul(A_train[layers - 2], masks[layers - 1] * W[layers - 1]) + B[layers - 1]
        A_train[layers - 1] = tf.nn.softmax(logits)
        A_test[layers - 1] = tf.nn.softmax(
            tf.matmul(A_test[layers - 2], masks[layers - 1] * W[layers - 1]) + B[layers - 1])
        Hypothesis_train = A_train[layers - 1]
        Hypothesis_test = A_test[layers - 1]
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(Y_train, logits))  # ADD: added "_v2" newVersion
    else:
        if function == 'tanh':
            A_train[layers - 1] = tf.tanh(
                tf.matmul(A_train[layers - 2], masks[layers - 1] * W[layers - 1]) + B[layers - 1])
            A_test[layers - 1] = tf.tanh(
                tf.matmul(A_test[layers - 2], masks[layers - 1] * W[layers - 1]) + B[layers - 1])
        else:
            A_train[layers - 1] = tf.sigmoid(
                tf.matmul(A_train[layers - 2], masks[layers - 1] * W[layers - 1]) + B[layers - 1])
            A_test[layers - 1] = tf.sigmoid(
                tf.matmul(A_test[layers - 2], masks[layers - 1] * W[layers - 1]) + B[layers - 1])
        Hypothesis_train = A_train[layers - 1]
        Hypothesis_test = A_test[layers - 1]
        loss = tf.reduce_mean(tf.squared_difference(Hypothesis_train, Y_train))

    train_step = tf.train.AdamOptimizer().minimize(loss)

    # Add an op to initialize the variables
    init = tf.initialize_all_variables()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    sess = tf.Session()
    # writer = tf.train.SummaryWriter("./logs/nn_logs", sess.graph)

    sess.run(init)

    t_start = time.clock()

    for i in range(iterations):
        x_train, y_train = x, y
        if batch_size != len(x):
            batch_indexes = random.sample(range(len(x_train)), batch_size)
            x_train = [e for (j, e) in enumerate(x) if j in batch_indexes]
            y_train = [e for (j, e) in enumerate(y) if j in batch_indexes]
        sess.run(train_step, feed_dict={X_train: x_train, Y_train: y_train, rate: 1})  # ADDED rate for TF v2
        if i % (iterations / 10) == 0:
            print('EPOCH ', i)
            print('COST ',
                  sess.run(loss, feed_dict={X_train: x_train, Y_train: y_train, rate: 1}))  # ADDED rate for TF v2
            h_train = sess.run(Hypothesis_train, feed_dict={X_train: x, rate: 1})  # ADDED rate for TF v2
            print('TRAIN ACCURACY', accuracy(x_train, y_train, h_train))
            h_test = sess.run(Hypothesis_test, feed_dict={X_test: x_test, rate: 1})  # ADDED rate for TF v2
            print('TEST ACCURACY', accuracy(x_test, y_test, h_test))
            if accuracy(x_test, y_test, h_test) == 1:
                break

    # Save the variables to disk
    save_path = saver.save(sess, 'models/' + model_name + '.ckpt')
    print("Model saved in file: %s" % save_path)

    t_end = time.clock()
    passed_time = 'Passed time: ' + str(t_end - t_start)
    print(passed_time)

    tf.reset_default_graph()

    h_train = sess.run(Hypothesis_train, feed_dict={X_train: x, rate: 1})  # ADDED rate for TF v2
    h_test = sess.run(Hypothesis_test, feed_dict={X_test: x_test, rate: 1})  # ADDED rate for TF v2
    train_acc = accuracy(x, y, h_train)
    test_acc = accuracy(x_test, y_test, h_test)
    return train_acc, test_acc


def execute_network(data, model_name, hidden_nodes, function='tanh', softmax=True):
    '''
	Executes a saved model, can be used to return parameters
	'''

    x_train, y_train = data.get_train_x_y()
    if data.num_vali == 0:
        x_vali, y_vali = x_train, y_train
    else:
        x_vali, y_vali = data.get_vali_x_y()
    if data.num_test == 0:
        x_test, y_test = x_train, y_train
    else:
        x_test, y_test = data.get_test_x_y()

    input_size = len(x_train[0])
    output_size = len(y_train[0])

    layers = len(hidden_nodes) + 1

    X_train = tf.placeholder(tf.float32, shape=[len(x_train), input_size])
    Y_train = tf.placeholder(tf.float32, shape=[len(x_train), output_size])
    X_vali = tf.placeholder(tf.float32, shape=[len(x_vali), input_size])
    Y_vali = tf.placeholder(tf.float32, shape=[len(x_vali), output_size])
    X_test = tf.placeholder(tf.float32, shape=[len(x_test), input_size])
    Y_test = tf.placeholder(tf.float32, shape=[len(x_test), output_size])

    # Initial weights and bias are set
    W = [None] * layers
    B = [None] * layers
    W[0] = init_weights([input_size, hidden_nodes[0]], 0)
    W[layers - 1] = init_weights([hidden_nodes[layers - 2], output_size], layers - 1)
    B[0] = init_bias(hidden_nodes[0], 0)
    B[layers - 1] = init_bias(output_size, layers - 1)
    for i in range(layers - 2):
        W[i + 1] = init_weights([hidden_nodes[i], hidden_nodes[i + 1]], i + 1)
        B[i + 1] = init_bias(hidden_nodes[i + 1], i + 1)

    # Lists that store the activation values
    A_train = [None] * layers
    A_vali = [None] * layers
    A_test = [None] * layers

    if function == 'tanh':
        A_train[0] = tf.tanh(tf.matmul(X_train, W[0]) + B[0])
        A_vali[0] = tf.tanh(tf.matmul(X_vali, W[0]) + B[0])
        A_test[0] = tf.tanh(tf.matmul(X_test, W[0]) + B[0])
    else:
        A_train[0] = tf.sigmoid(tf.matmul(X_train, W[0]) + B[0])
        A_vali[0] = tf.sigmoid(tf.matmul(X_vali, W[0]) + B[0])
        A_test[0] = tf.sigmoid(tf.matmul(X_test, W[0]) + B[0])

    for i in range(1, layers - 1):
        if function == 'tanh':
            A_train[i] = tf.tanh(tf.matmul(A_train[i - 1], W[i]) + B[i])
            A_vali[i] = tf.tanh(tf.matmul(A_vali[i - 1], W[i]) + B[i])
            A_test[i] = tf.tanh(tf.matmul(A_test[i - 1], W[i]) + B[i])
        else:
            A_train[i] = tf.sigmoid(tf.matmul(A_train[i - 1], W[i]) + B[i])
            A_vali[i] = tf.sigmoid(tf.matmul(A_vali[i - 1], W[i]) + B[i])
            A_test[i] = tf.sigmoid(tf.matmul(A_test[i - 1], W[i]) + B[i])

    if softmax:
        # Softmax layer
        logits = tf.matmul(A_train[layers - 2], W[layers - 1]) + B[layers - 1]
        A_train[layers - 1] = tf.nn.softmax(logits)
        A_vali[layers - 1] = tf.nn.softmax(tf.matmul(A_vali[layers - 2], W[layers - 1]) + B[layers - 1])
        A_test[layers - 1] = tf.nn.softmax(tf.matmul(A_test[layers - 2], W[layers - 1]) + B[layers - 1])
    else:
        if function == 'tanh':
            A_train[layers - 1] = tf.tanh(tf.matmul(A_train[layers - 2], W[layers - 1]) + B[layers - 1])
            A_vali[layers - 1] = tf.tanh(tf.matmul(A_vali[layers - 2], W[layers - 1]) + B[layers - 1])
            A_test[layers - 1] = tf.tanh(tf.matmul(A_test[layers - 2], W[layers - 1]) + B[layers - 1])
        else:
            A_train[layers - 1] = tf.sigmoid(tf.matmul(A_train[layers - 2], W[layers - 1]) + B[layers - 1])
            A_vali[layers - 1] = tf.sigmoid(tf.matmul(A_vali[layers - 2], W[layers - 1]) + B[layers - 1])
            A_test[layers - 1] = tf.sigmoid(tf.matmul(A_test[layers - 2], W[layers - 1]) + B[layers - 1])

    Hypothesis_train = A_train[layers - 1]
    Hypothesis_vali = A_vali[layers - 1]
    Hypothesis_test = A_test[layers - 1]

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    sess = tf.Session()
    # writer = tf.train.SummaryWriter("./logs/nn_logs", sess.graph)

    # Restore trained network
    saver.restore(sess, 'models/' + model_name + '.ckpt')
    print("Model " + model_name + " restored")

    t_start = time.clock()

    h_train = sess.run(Hypothesis_train, feed_dict={X_train: x_train})
    acc = accuracy(x_train, y_train, h_train)
    print('TRAIN ACCURACY', acc)
    h_vali = sess.run(Hypothesis_vali, feed_dict={X_vali: x_vali})  # auskommentiert
    print('VALIDATION ACCURACY', accuracy(x_vali, y_vali, h_vali))  # ...
    h_test = sess.run(Hypothesis_test, feed_dict={X_test: x_test})
    print('TEST ACCURACY', accuracy(x_test, y_test, h_test))

    activation_values_train = [None] * layers
    activation_values_vali = [None] * layers
    activation_values_test = [None] * layers

    weights = [None] * layers
    bias = [None] * layers

    for j in range(layers - 1):
        activation_values_train[j] = sess.run(A_train[j], feed_dict={X_train: x_train})
        activation_values_vali[j] = sess.run(A_vali[j], feed_dict={X_vali: x_vali})
        activation_values_test[j] = sess.run(A_test[j], feed_dict={X_test: x_test})
        weights[j] = sess.run(W[j])
        bias[j] = sess.run(B[j])
    activation_values_train[layers - 1] = np.asarray(
        one_hot_code(sess.run(Hypothesis_train, feed_dict={X_train: x_train})))
    activation_values_vali[layers - 1] = np.asarray(one_hot_code(sess.run(Hypothesis_vali, feed_dict={X_vali: x_vali})))
    activation_values_test[layers - 1] = np.asarray(one_hot_code(sess.run(Hypothesis_test, feed_dict={X_test: x_test})))
    weights[layers - 1] = sess.run(W[layers - 1])
    bias[layers - 1] = sess.run(B[layers - 1])

    t_end = time.clock()
    passed_time = 'Passed time: ' + str(t_end - t_start)
    print(passed_time)

    tf.reset_default_graph()

    return (activation_values_train, activation_values_vali, activation_values_test, weights, bias, acc)


# return accuracy(x_test, y_test, h_test)

def weight_sparseness_pruning(data, model_name, new_model_name, hidden_nodes, iterations, function='tanh', softmax=True,
                              accuracy_decrease=0.01, indexes=[], to_prune=[], together=1, batch_size=0,
                              max_non_improvements=2, max_runs=5):
    '''
	param data: a DataSet instance
	param model_name: name with which model will be stored
	param hidden_nodes: number of nodes on each hidden layer, as [[3], [4], [4]] for a network with three hidden layers
	param iterations: iterations for retraining
	param softmax: softmax layer at the end?
	param accuracy_decrease: allowed accuracy decrease for a connection to remain pruned
	param together: number of connections that are pruned together
	param batch_size: size of a training batch, '0' means no batch training
	param max_non_improvements: after pruning some index, number of times a retraining iteration set is performed without 
	increase of accuracy before the connection is deemed not prunable
	param max_runs: after pruning some index, number of times a retraining iteration set is performed before the connection 
	is deemed not prunable
	'''

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
        return functools.reduce(concat, [[(h, i, j) for (i, j) in
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
                rows[(m, pr)] = sum(1 for (i, j) in indexes[m] if i == pr)
            poss_cols = range(shapes[m][1])
            for pc in poss_cols:
                cols[(m, pc)] = sum(1 for (i, j) in indexes[m] if j == pc)
        return rows, cols

    x, y = data.get_train_x_y()
    if data.num_vali == 0:
        x_vali, y_vali = data.get_train_x_y()
    else:
        x_vali, y_vali = data.get_vali_x_y()

    input_size = len(x[0])
    output_size = len(y[0])

    layers = len(hidden_nodes) + 1

    if batch_size == 0:
        batch_size = len(x)

    X_train = tf.placeholder(tf.float32, shape=[batch_size, input_size])
    Y_train = tf.placeholder(tf.float32, shape=[batch_size, output_size])
    X_vali = tf.placeholder(tf.float32, shape=[len(x_vali), input_size])
    Y_vali = tf.placeholder(tf.float32, shape=[len(x_vali), output_size])

    W = [None] * layers
    masks = [None] * layers
    B = [None] * layers

    W[0] = init_weights([input_size, hidden_nodes[0]], 0)
    masks[0] = tf.placeholder(tf.float32, shape=[input_size, hidden_nodes[0]])
    W[layers - 1] = init_weights([hidden_nodes[layers - 2], output_size], layers - 1)
    masks[layers - 1] = tf.placeholder(tf.float32, shape=[hidden_nodes[layers - 2], output_size])
    B[0] = init_bias(hidden_nodes[0], 0)
    B[layers - 1] = init_bias(output_size, layers - 1)

    for i in range(layers - 2):
        W[i + 1] = init_weights([hidden_nodes[i], hidden_nodes[i + 1]], i + 1)
        B[i + 1] = init_bias(hidden_nodes[i + 1], i + 1)
        masks[i + 1] = tf.placeholder(tf.float32, shape=[hidden_nodes[i], hidden_nodes[i + 1]])

    # Lists that store the activation values
    A_train = [None] * layers
    A_vali = [None] * layers

    if function == 'tanh':
        A_train[0] = tf.tanh(tf.matmul(X_train, masks[0] * W[0]) + B[0])
        A_vali[0] = tf.tanh(tf.matmul(X_vali, masks[0] * W[0]) + B[0])
    else:
        A_train[0] = tf.sigmoid(tf.matmul(X_train, masks[0] * W[0]) + B[0])
        A_vali[0] = tf.sigmoid(tf.matmul(X_vali, masks[0] * W[0]) + B[0])

    for i in range(1, layers - 1):
        if function == 'tanh':
            A_train[i] = tf.tanh(tf.matmul(A_train[i - 1], masks[i] * W[i]) + B[i])
            A_vali[i] = tf.tanh(tf.matmul(A_vali[i - 1], masks[i] * W[i]) + B[i])
        else:
            A_train[i] = tf.sigmoid(tf.matmul(A_train[i - 1], masks[i] * W[i]) + B[i])
            A_vali[i] = tf.sigmoid(tf.matmul(A_vali[i - 1], masks[i] * W[i]) + B[i])

    # Softmax layer
    logits = tf.matmul(A_train[layers - 2], masks[layers - 1] * W[layers - 1]) + B[layers - 1]
    A_train[layers - 1] = tf.nn.softmax(logits)
    A_vali[layers - 1] = tf.nn.softmax(tf.matmul(A_vali[layers - 2], masks[layers - 1] * W[layers - 1]) + B[layers - 1])
    Hypothesis_train = A_train[layers - 1]
    Hypothesis_vali = A_vali[layers - 1]

    if softmax:
        # Softmax layer
        logits = tf.matmul(A_train[layers - 2], masks[layers - 1] * W[layers - 1]) + B[layers - 1]
        A_train[layers - 1] = tf.nn.softmax(logits)
        A_vali[layers - 1] = tf.nn.softmax(
            tf.matmul(A_vali[layers - 2], masks[layers - 1] * W[layers - 1]) + B[layers - 1])
        Hypothesis_train = A_train[layers - 1]
        Hypothesis_vali = A_vali[layers - 1]
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits, Y_train))
    else:
        if function == 'tanh':
            A_train[layers - 1] = tf.tanh(
                tf.matmul(A_train[layers - 2], masks[layers - 1] * W[layers - 1]) + B[layers - 1])
            A_vali[layers - 1] = tf.tanh(
                tf.matmul(A_vali[layers - 2], masks[layers - 1] * W[layers - 1]) + B[layers - 1])
        else:
            A_train[layers - 1] = tf.sigmoid(
                tf.matmul(A_train[layers - 2], masks[layers - 1] * W[layers - 1]) + B[layers - 1])
            A_vali[layers - 1] = tf.sigmoid(
                tf.matmul(A_vali[layers - 2], masks[layers - 1] * W[layers - 1]) + B[layers - 1])
        Hypothesis_train = A_train[layers - 1]
        Hypothesis_vali = A_vali[layers - 1]
        loss = tf.reduce_mean(tf.squared_difference(Hypothesis_train, Y_train))

    train_step = tf.train.AdamOptimizer().minimize(loss)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    sess = tf.Session()
    # writer = tf.train.SummaryWriter("./logs/nn_logs", sess.graph)

    # Restore trained network
    # saver.restore(sess, model_name)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, 'models/' + model_name + '.ckpt')
    print("Model " + model_name + " restored")
    t_start = time.clock()

    # If no indexes are specified, start by not pruning any indexes
    if not indexes:
        indexes = [None] * layers
        for i in range(layers):
            indexes[i] = []

    # If it is not specified what layers to prune, prune all layers (except for softmax if present)
    if not to_prune:
        if softmax:
            to_prune = range(layers - 1)
        else:
            to_prune = range(layers)

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
    target_accuracy = initial_vali_acc - accuracy_decrease
    print('Target accuracy: ', target_accuracy)
    new_accuracy = target_accuracy

    # Initial parameters are stored
    weights = [None] * layers
    bias = [None] * layers
    for j in range(layers):
        weights[j] = sess.run(W[j])
        bias[j] = sess.run(B[j])

    while together > 0:
        # print('together', together)
        print('Left indexes: ' + str(len(left_indexes)))
        pos_ind = left_indexes
        left_indexes = []
        print(pos_ind)
        print(row_count)
        print(col_count)
        # While there are still possible indexes to prune
        while pos_ind:
            # Sort possible indexes in order of increasing pruned connections
            # pos_ind.sort(key=lambda m,i,j:row_count[(m, i)] + col_count[(m, j)])
            # sorted_poss_ind = sorted(pos_ind, key=lambda m, i, j:row_count[(m, i)] + col_count[(m, j)])
            # pos_ind = sorted_poss_ind
            non_found = True
            for i in range(len(pos_ind)):
                selected_indexes = []
                for index in range(together):
                    selected_indexes.append(pos_ind.pop(0))
                print('Pruned indexies: ' + str(sum(len(l) for l in indexes)))
                print('Remaining indexies: ' + str(len(pos_ind)))
                print('left for next: ' + str(len(left_indexes)))
                for (m, r, c) in selected_indexes:
                    indexes[m].append((r, c))
                for mat in set(m for (m, r, c) in selected_indexes):
                    current_masks[mat] = remake_mask(shapes[mat], indexes[mat])

                for j in range(layers):
                    assign_op = W[j].assign(W[j] * tf.constant(current_masks[j], dtype=tf.float32))
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
                            batch_indexes = random.sample(range(len(x_train)), batch_size)
                            x_train = [e for (j, e) in enumerate(x) if j in batch_indexes]
                            y_train = [e for (j, e) in enumerate(y) if j in batch_indexes]
                        placeholder_dict.update({X_train: x_train, Y_train: y_train})
                        sess.run(train_step, feed_dict=placeholder_dict)
                        # for j in range(layers):
                        #	assign_op = W[j].assign(W[j]*tf.constant(current_masks[j], dtype=tf.float32))
                        #	sess.run(assign_op)
                        h_vali = sess.run(Hypothesis_vali, feed_dict=placeholder_dict)
                        new_accuracy = accuracy(x_vali, y_vali, h_vali)
                        if i % (iterations / 2) == 0:
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
                        # for j in range(layers):
                        #	assign_op = W[j].assign(W[j]*tf.constant(current_masks[j], dtype=tf.float32))
                        #	sess.run(assign_op)

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
                        indexes[m].remove((r, c))

                    for mat in set(m for (m, r, c) in selected_indexes):
                        current_masks[mat] = remake_mask(shapes[mat], indexes[mat])

                    # del indexes[m][-1]
                    # current_masks[m] = remake_mask(shapes[m], indexes[m])

                    placeholder_dict.update({i: d for i, d in zip(masks, current_masks)})

                    for j in range(layers):
                        assign_op = W[j].assign(
                            tf.constant(weights[j], dtype=tf.float32) * tf.constant(current_masks[j], dtype=tf.float32))
                        sess.run(assign_op)
                        assign_op = B[j].assign(tf.constant(bias[j], dtype=tf.float32))
                        sess.run(assign_op)
                    h_vali = sess.run(Hypothesis_vali, feed_dict=placeholder_dict)
                    acc = accuracy(x_vali, y_vali, h_vali)
                    print('Accuracy: ', acc)
                else:
                    print('Index found, break from for')
                    break
            # If no index could be pruned, break out of while loop
            if non_found:
                print('Breaking from while, no index can be pruned')
                break
        together = int(float(together) / 2)

    # Save the variables to disk
    save_path = saver.save(sess, 'models/' + new_model_name + '.ckpt')
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


def rexren_input_prune(data, model_name, new_model_name, hidden_nodes, function, softmax, max_accuracy_decrease=5,
                       per_threshold=True):
    '''
	Performs RxREN pruning of inputs

	param data: a DataSet instance
	param model_name: name with which model will be stored
	param hidden_nodes: number of nodes on each hidden layer, as [[3], [4], [4]] for a network with three hidden layers
	param function: activation function, 'tanh' or 'sigmoid', but can be adapted for other functions
	param softmax: softmax layer at the end?	
	param max_acuracy_decrease: maximum allowed percentual accuracy decrease
	param per_threshold: The algorithm attempt to prune all inputs with the 
	same degree of irrelevance at once, either they are all pruned or none is
	'''

    x_train, y_train = data.get_train_x_y()
    if data.num_vali == 0:
        x_vali, y_vali = x_train, y_train
    else:
        x_vali, y_vali = data.get_vali_x_y()

    input_size = len(x_train[0])
    output_size = len(y_train[0])

    layers = len(hidden_nodes) + 1

    X_train = tf.placeholder(tf.float32, shape=[len(x_train), input_size])
    Y_train = tf.placeholder(tf.float32, shape=[len(x_train), output_size])
    X_vali = tf.placeholder(tf.float32, shape=[len(x_vali), input_size])
    Y_vali = tf.placeholder(tf.float32, shape=[len(x_vali), output_size])

    # Initial weights and bias are set
    W = [None] * layers
    B = [None] * layers

    W[0] = init_weights([input_size, hidden_nodes[0]], 0)
    mask = tf.placeholder(tf.float32, shape=[input_size, hidden_nodes[0]])
    W[layers - 1] = init_weights([hidden_nodes[layers - 2], output_size], layers - 1)
    B[0] = init_bias(hidden_nodes[0], 0)
    B[layers - 1] = init_bias(output_size, layers - 1)

    for i in range(layers - 2):
        W[i + 1] = init_weights([hidden_nodes[i], hidden_nodes[i + 1]], i + 1)
        B[i + 1] = init_bias(hidden_nodes[i + 1], i + 1)

    # Lists that store the activation values
    A_train = [None] * layers
    A_vali = [None] * layers
    masked_weights = mask * W[0]

    if function == 'tanh':
        A_train[0] = tf.tanh(tf.matmul(X_train, masked_weights) + B[0])
        A_vali[0] = tf.tanh(tf.matmul(X_vali, masked_weights) + B[0])
    else:
        A_train[0] = tf.sigmoid(tf.matmul(X_train, masked_weights) + B[0])
        A_vali[0] = tf.sigmoid(tf.matmul(X_vali, masked_weights) + B[0])

    for i in range(1, layers - 1):
        if function == 'tanh':
            A_train[i] = tf.tanh(tf.matmul(A_train[i - 1], W[i]) + B[i])
            A_vali[i] = tf.tanh(tf.matmul(A_vali[i - 1], W[i]) + B[i])
        else:
            A_train[i] = tf.sigmoid(tf.matmul(A_train[i - 1], W[i]) + B[i])
            A_vali[i] = tf.sigmoid(tf.matmul(A_vali[i - 1], W[i]) + B[i])

    if softmax:
        # Softmax layer
        logits = tf.matmul(A_train[layers - 2], W[layers - 1]) + B[layers - 1]
        A_train[layers - 1] = tf.nn.softmax(logits)
        A_vali[layers - 1] = tf.nn.softmax(tf.matmul(A_vali[layers - 2], W[layers - 1]) + B[layers - 1])
    else:
        if function == 'tanh':
            A_train[layers - 1] = tf.tanh(tf.matmul(A_train[layers - 2], W[layers - 1]) + B[layers - 1])
            A_vali[layers - 1] = tf.tanh(tf.matmul(A_vali[layers - 2], W[layers - 1]) + B[layers - 1])
        else:
            A_train[layers - 1] = tf.sigmoid(tf.matmul(A_train[layers - 2], W[layers - 1]) + B[layers - 1])
            A_vali[layers - 1] = tf.sigmoid(tf.matmul(A_vali[layers - 2], W[layers - 1]) + B[layers - 1])

    Hypothesis_train = A_train[layers - 1]
    Hypothesis_vali = A_vali[layers - 1]

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    sess = tf.Session()
    # writer = tf.train.SummaryWriter("./logs/nn_logs", sess.graph)

    # Restore trained network
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, 'models/' + model_name + '.ckpt')
    print("Model " + model_name + " restored")
    t_start = time.clock()

    pruned_inputs = []
    accepted_pruned_indexes = []
    to_prune = []

    indexes = []
    new_indexes = []
    shape = sess.run(W[0]).shape
    current_mask = remake_mask(shape, indexes)

    placeholder_dict = {X_train: x_train, Y_train: y_train, X_vali: x_vali, Y_vali: y_vali, mask: current_mask}

    h_vali = sess.run(Hypothesis_vali, feed_dict=placeholder_dict)
    vali_acc = accuracy(x_vali, y_vali, h_vali)
    # The minimum allowed accuracy is calculated
    min_allowed_accuracy = vali_acc - (vali_acc * float(max_accuracy_decrease) / 100)
    print('Minimum allowed accuracy: ', min_allowed_accuracy)
    # As long as the new validation accuracy does not fall below it, inputs are pruned
    while vali_acc >= min_allowed_accuracy and len(accepted_pruned_indexes) < input_size:
        accepted_pruned_indexes += new_indexes
        # List of (err(i), i, [indexes]) tuples is initialized
        input_errors = []
        # The inputs selected as least relevant on the previous step are pruned
        pruned_inputs += to_prune
        print('pruned_inputs: ', pruned_inputs)
        to_prune = []
        # The new classification of the training set is calculated
        h_train = sess.run(Hypothesis_train, feed_dict=placeholder_dict)
        correct, misses = classification_example_indexes(x_train, y_train, h_train)
        for i in [j for j in range(input_size) if j not in pruned_inputs]:
            indexes = accepted_pruned_indexes[:]
            new_indexes = indexes_of_neuron(shape, i)
            indexes += new_indexes
            current_mask = remake_mask(shape, indexes)
            placeholder_dict[mask] = current_mask
            h_train = sess.run(Hypothesis_train, feed_dict=placeholder_dict)
            i_correct, i_misses = classification_example_indexes(x_train, y_train, h_train)
            i_misses = i_misses.intersection(correct)
            input_errors.append((len(i_misses), i, new_indexes))
        indexes = accepted_pruned_indexes[:]
        new_indexes = []
        # If 'per_threshold' is set, all inputs with the least significance
        # value are pruned at the same time.
        if per_threshold:
            most_irrelevant = [x for x in input_errors if x[0] == min(input_errors)[0]]
            for i in most_irrelevant:
                to_prune.append(i[1])
                new_indexes += i[2]
        else:
            most_irrelevant = min(input_errors)
            to_prune.append(most_irrelevant[1])
            new_indexes += most_irrelevant[2]
        indexes += new_indexes
        current_mask = remake_mask(shape, indexes)
        placeholder_dict[mask] = current_mask
        h_vali = sess.run(Hypothesis_vali, feed_dict=placeholder_dict)
        vali_acc = accuracy(x_vali, y_vali, h_vali)
        print('vali_acc', vali_acc)
    current_mask = remake_mask(shape, accepted_pruned_indexes)
    placeholder_dict[mask] = current_mask
    h_vali = sess.run(Hypothesis_vali, feed_dict=placeholder_dict)
    vali_acc = accuracy(x_vali, y_vali, h_vali)
    print('Final accuracy: ', vali_acc)
    print('Pruned inputs: ', pruned_inputs)
    weights = sess.run(W[0], feed_dict=placeholder_dict)
    assign_op = W[0].assign(tf.constant(weights, dtype=tf.float32) * tf.constant(current_mask, dtype=tf.float32))
    sess.run(assign_op)

    # Save the variables to disk
    save_path = saver.save(sess, 'models/' + new_model_name + '.ckpt')
    print("Model saved in file: %s" % save_path)

    t_end = time.clock()
    passed_time = 'Passed time: ' + str(t_end - t_start)
    print(passed_time)

    tf.reset_default_graph()
