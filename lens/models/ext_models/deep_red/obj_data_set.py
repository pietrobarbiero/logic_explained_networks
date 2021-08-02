import csv
import os

import numpy as np
from . import load_restore as lr


# A dataset is a collection of observations. Each observation is assigned
# an index. From that dataset, a subset such as a training, testing or 
# validation set is defined by using a subset of indexes.

class DataSet:
    def __init__(self, dataset_name, hidden_nodes, input_columns=None, output_column=None):
        '''
		Initializes a dataset
		'''
        file_name = os.path.join("data", dataset_name + '.csv')
        # file_name = dataset_name + '.csv'
        with open(file_name, 'r') as f:
            data_iter = csv.reader(f, delimiter=',')
            data = [[float(attr) for attr in data] for data in data_iter]
        # If the input and output columns are not defined, then all columns
        # except the last are the input and the last column is the class value
        if not input_columns:
            input_columns = range(len(data[0]) - 1)
        if not output_column:
            # Index of class value in the data
            output_column = len(data[0]) - 1
        # If no classes are defined for the output column, these are 0 or 1
        # (it is a binary classification problem)
        self.classes = sorted(set(int(d[-1]) for d in data))
        self.number_examples = len(data)
        self.output_neurons = len(self.classes)
        self.network_length = len(hidden_nodes) + 2
        self.output_layer = len(hidden_nodes) + 1
        self.input_lenght = len(input_columns)
        self.examples = [None] * self.number_examples

        def to_one_hot(value, bits):
            one_hot = [0] * bits
            one_hot[value] = 1
            return one_hot

        for e in range(self.number_examples):
            self.examples[e] = Observation(e, self.network_length,
                                           int(data[e][output_column]),
                                           np.asarray(list(data[e][i] for i in input_columns)),
                                           to_one_hot(int(data[e][output_column]), self.output_neurons))

    def set_split(self, train_indexes, vali_indexes, test_indexes):
        '''
		Determines which instances make out the train, validation and test sets
		'''
        self.train_indexes = train_indexes
        self.vali_indexes = vali_indexes
        self.test_indexes = test_indexes
        self.num_train = len(train_indexes)
        self.num_vali = len(vali_indexes)
        self.num_test = len(test_indexes)

    def initialize_dictionary(self, output_condition):
        '''
		Initializes a dictionary that stores for each split point the indexes
		from the train and validation sets that exceed the threshold of that 
		split-point.
		'''
        self.example_cond_dict = {}
        # If there is no separate validation set, then the post-pruning is performed on the train set
        # if self.num_vali > 0:
        #	self.dict_indexes = self.vali_indexes
        # else:
        self.dict_indexes = self.train_indexes
        self.update_dictionary([(output_condition[0], output_condition[1], output_condition[2])])
        # The target class value is stored under the same key as the output split point, but with layer index -1
        self.example_cond_dict[(-1, output_condition[1], output_condition[2])] = [e for e in self.dict_indexes if
                                                                                  self.examples[e].class_value ==
                                                                                  output_condition[1]]

    def update_dictionary(self, split_points):
        dictionary = self.example_cond_dict
        indexes = self.dict_indexes
        sps = set(split_points).difference(dictionary.keys())
        for sp in sps:
            dictionary[sp] = [e for e in indexes if self.examples[e].values[sp[0]][sp[1]] > sp[2]]

    def get_train_x_y(self):
        x = [self.examples[i].values[0] for i in self.train_indexes]
        y = [self.examples[i].y for i in self.train_indexes]
        return (x, y)

    def get_vali_x_y(self):
        x = [self.examples[i].values[0] for i in self.vali_indexes]
        y = [self.examples[i].y for i in self.vali_indexes]
        return (x, y)

    def get_test_x_y(self):
        x = [self.examples[i].values[0] for i in self.test_indexes]
        y = [self.examples[i].y for i in self.test_indexes]
        return (x, y)

    def set_train_act_values(self, layer, values=[]):
        for i in range(self.num_train):
            e = self.train_indexes[i]
            self.examples[e].set_act_values(layer, values[i])

    def set_vali_act_values(self, layer, values=[]):
        for i in range(self.num_vali):
            e = self.vali_indexes[i]
            self.examples[e].set_act_values(layer, values[i])

    def set_test_act_values(self, layer, values=[]):
        for i in range(self.num_test):
            e = self.test_indexes[i]
            self.examples[e].set_act_values(layer, values[i])

    def set_act_values(self, act_train, act_vali, act_test):
        for i in range(self.network_length - 1):
            if self.num_train > 0:
                self.set_train_act_values(i + 1, act_train[i])
            if self.num_vali > 0:
                self.set_vali_act_values(i + 1, act_vali[i])
            if self.num_test > 0:
                self.set_test_act_values(i + 1, act_test[i])

    def get_train_vali_obs(self):
        return [self.examples[e] for e in self.train_indexes] + [self.examples[e] for e in self.vali_indexes]

    def get_train_obs(self):
        return [self.examples[e] for e in self.train_indexes]

    def get_vali_obs(self):
        return [self.examples[e] for e in self.vali_indexes]

    def get_test_obs(self):
        return [self.examples[e] for e in self.test_indexes]

    # returns a list of all activation values for a node
    def get_act_all_examples(self, layer, node):
        vals = [e.values[layer][node] for e in self.examples if e.idx in self.train_indexes or e in self.vali_indexes]
        return vals


class Observation:
    def __init__(self, idx, network_length, class_value, x=[], y=[]):
        self.idx = idx
        self.values = [None] * network_length
        self.values[0] = x  # Layer with index 0 is the input
        self.y = y
        self.class_value = class_value
        self.orig_prediction = 0

    def set_nn_prediction(self, value):
        self.orig_prediction = value

    #  A 'split_point' has the form (layer, neuron, threshold)
    # The method returns True if >, False if <=
    def side_of_threshold(self, split_point):
        return self.values[split_point[0]][split_point[1]] > split_point[2]

    # A 'condition' has the form (layer, neuron, threshold, operator),
    # where operator is 0 for <=', 1 for '>' and 2 for '='
    def fulfills_cond(self, condition):
        if isinstance(condition, tuple):
            bigger = self.values[condition[0]][condition[1]] > condition[2]
            if condition[3]:
                if bigger:
                    return True
                else:
                    return False
            else:
                if bigger:
                    return False
                else:
                    return True
        else:
            return condition

    def fulfills_rule(self, rule):
        if isinstance(rule, list):
            if all((self.fulfills_cond(c) for c in rule)):
                return True
            else:
                return False
        else:
            return rule

    def fulfills_dnf(self, dnf):
        if isinstance(dnf, list):
            if any((self.fulfills_rule(r) for r in dnf)):
                return True
            else:
                return False
        else:
            return dnf

    def set_act_values(self, layer, values=[]):
        self.values[layer] = values
