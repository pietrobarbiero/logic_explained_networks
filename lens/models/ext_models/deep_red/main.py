import math
import os
import sys
import time

import numpy as np

from lens.models.ext_models.deep_red import decision_tree_induction as dti
from lens.models.ext_models.deep_red import deep_nn_train as dnnt
from lens.models.ext_models.deep_red import evaluation_formulas as ef
from lens.models.ext_models.deep_red import load_restore as lr
from lens.models.ext_models.deep_red import printer
from lens.models.ext_models.deep_red import replacement as r
from lens.models.ext_models.deep_red import split_determinator as sd
from lens.models.ext_models.deep_red.obj_data_set import DataSet


# Determine one or more splits of train and test data. Note that
# different splits can be used to train the networks and extract the rule 
# models (for instance a subset of the train data can be used to extract 
# the rule model).

def set_split_manually(dataset_name, split_name, train_indexes, test_indexes):
    '''
	Manually select which indexes will be used for training and which
	for testing.

	param dataset_name: name of dataset without .csv
	param split_name: name that will be assigned to split
	param train: list of indexes that will be used for training
	param test: list of indexes that will be used for testing
	'''
    lr.save_train_indexes(train_indexes, dataset_name, split_name)
    lr.save_test_indexes(test_indexes, dataset_name, split_name)


def set_split(dataset_name, split_name, percentage):
    '''
	Splits data in train and test sets randomly maintaining
	the class distribution and saves the corresponding
	indexes in .pkl files in the 'indexes' folder.

	param dataset_name: name of dataset without .csv
	param split_name: name that will be assigned to split
	param percentage: percentage of instances that used for training
	'''
    sd.initial_splits(dataset_name, split_name, percentage)


def set_cv_folds(dataset_name, k):
    """
	Split data into different folds. The resulting split names have the
	form cv_<k>-<i>, where i is the current folds used for testing.

	param dataset_name: name of dataset without .csv
	param k: number of folds
	"""
    return sd.cross_validation_folds(dataset_name, k, return_names=True)


# If model not present, train a neural network using dnnt.train_network. 
#
# Retrain to perform WSP/RxREN pruning with dnnt.weight_sparseness_pruning
# or dnnt.rexren_input_prune, repectively. Only sigmoid or hyperbolic tangent 
# activation functions can be used, but this can be easily adapted.
#
# Retrain to perform activation polarization with ktp.retrain_network, or
# follow this with WSP with ktp.keep_training_wsp_polarize. Currently, this
# only works with the hyperbolic tangent activationfunction

def prepare_network(dataset_name, split_name, model_name, hidden_nodes,
                    init_iterations=1000, wsp_iterations=100, wsp_accuracy_decrease=0.02,
                    rxren_accuracy_decrease=5, function='tanh', softmax=True):
    """
	param dataset_name: name of dataset without .csv
	param split_name: name of the split
	param model_name: how the model will be stored
	param hidden_nodes: number of nodes on each hidden layer, as [[3], [4], [4]] for a network with three hidden layers
	param init_iterations: initial number of iterations for training
	param wsp_iterations: itarations WSP uses for each retraining step
	param wsp_accuracy_decrease: allowed accuracy decrease for WSP
	param rxren_accuracy_decrease: allowed accuracy decrease for RxREN
	param function: activation function, 'tanh' or 'sigmoid'
	param softmax: softmax layer at the end?
	"""
    train, test = lr.load_indexes(dataset_name, split_name)
    data = DataSet(dataset_name, hidden_nodes)
    data.set_split(train, [], test)  # data.set_split(train, test, test)
    train_acc, test_acc = dnnt.train_network(data, model_name, hidden_nodes,
                                             iterations=init_iterations, function=function, softmax=softmax)

    # ktp.retrain_network(data, model_name, model_name+'pol', hidden_nodes, int(init_iterations/10))

    # dnnt.weight_sparseness_pruning(data, model_name, model_name, hidden_nodes, iterations=wsp_iterations, function=function, softmax=softmax, accuracy_decrease=wsp_accuracy_decrease)

    # dnnt.rexren_input_prune(data, model_name, model_name, hidden_nodes, function=function, softmax=softmax, max_accuracy_decrease=rxren_accuracy_decrease)

    return train_acc, test_acc


# Extract the rule set model

def extract_model(dataset_name, split_name, model_name, hidden_nodes,
                  target_class_index, function='tanh', softmax=True, class_dominance=98,
                  min_set_size=1, dis_config=0, rft_pruning_config=1, rep_pruning_config=1,
                  print_excel_results=False):
    '''
	param dataset_name: name of dataset without .csv
	param split_name: name of the split
	param model_name: name of the network model
	param hidden_nodes: number of nodes on each hidden layer, as
		[[3], [4], [4]] for a network with three hidden layers
	param target_class_index: class for which rules will be extracted
	param class_dominance: a percentage of the data set size on a branch
		If that number of examples are classified correctly without further
		increasing the tree, it stops growing
	param min_set_size:  a percentage of the initial training set size.
		If the dataset on a branch is smaller than that number, the tree
		stops growing
	param dis_config: discretization configuration for the thresholds
		that divide each neuron's activation range.
	param rft_pruning_config: post-pruning of intermediate expressions,
		2 is with, 1 is without
	param rep_pruning_config: post-pruning during replacement steps,
		2 is with, 1 is without
	param print_excel_results: prints some sheets of information in Excel
		about the extracted models
	'''
    # Standard output condition. Note that this isn't treated as the output
    # neuron of that class exceeding threshold 0.5 but as that observation being
    # predicted to be of that class
    print('set output_condition')
    output_condition = (len(hidden_nodes) + 1, target_class_index, 0.5, True)
    print('set output_condition finished')

    # Build dataset
    print('build dataset')
    data = DataSet(dataset_name, hidden_nodes)
    print('build dataset finished')

    # Set split
    print('set splits')
    train, test = lr.load_indexes(dataset_name, split_name)
    data.set_split(train, [], test)
    print('set splits finished')

    # Get activation values and parameters
    print('execute network')
    act_train, _, act_test, weights, _, _ = dnnt.execute_network(data, model_name, hidden_nodes, function=function,
                                                                 softmax=softmax)
    data.set_act_values(act_train, [], act_test)
    print('execute network finished')

    # Determine what neurons are relevant
    print('relevant neurons dict')
    rel_neuron_dict = dti.relevant_neurons(weights, hidden_nodes, data.input_lenght, output_len=data.output_neurons)
    print('relevant neurons dict finished')

    # Initialize condition example dictionary
    print('initialize dic')
    data.initialize_dictionary(output_condition)
    print('initialize dic finished')

    # Determine fixes min size
    min_size = math.ceil(float(len(data.dict_indexes)) * min_set_size / 100)

    # Extract a dictionary which links conditions of layer l with a dnf
    # using conditions of layer l-1 (and saves it to the 'obj' folder)
    bnn_name = f"{dataset_name}_{split_name}_{target_class_index}"
    print('BNN extraction')
    if os.path.exists('obj/BNN_' + bnn_name + '.pkl'):
        BNN, data.example_cond_dict, data.dict_indexes = lr.load_BNN_ecd_indexes(bnn_name)
        print('\nLoaded BNN, example-condition-dict, indexes')
    else:
        t = time.time()
        BNN = dti.build_BNN(data, output_condition, cd=class_dominance, mss=min_size,
                            relevant_neuron_dictionary=rel_neuron_dict, with_data=rft_pruning_config,
                            discretization=dis_config, cluster_means=None)
        lr.save_BNN_ecd_indexes(BNN, data.example_cond_dict, data.dict_indexes, bnn_name)
        print('\nBuilt BNN')
        print('Time: ', time.time() - t)
        print(BNN)
        print('BNN extraction finished')

    bio_name = f"{dataset_name}_{split_name}_{target_class_index}"
    # Extract an expression of an output condition w.r.t the inputs
    if os.path.exists('obj/bio_' + bio_name + '.pkl'):
        bio = lr.load_bio(bio_name)
        print('\nLoaded bio')
    else:
        t = time.time()
        bio = r.get_bio(BNN, output_condition, data.example_cond_dict, data.dict_indexes, with_data=rep_pruning_config,
                        data=data)
        lr.save_bio(bio, bio_name)
        print('\nBuilt bio')
        print('Time: ', time.time() - t)

    if isinstance(bio, list):
        complexity = sum(len(r) for r in bio)
        print('Number rules:', len(bio))
        print('Number terms:', complexity)
    else:
        complexity = 1

    fidelity = ef.accuracy_of_dnf(data, output_condition, bio, True, False, False, True)
    exp_accuracy = ef.class_accuracy(data, bio, target_class_index, False, True, False, True)
    print('Fidelity:', fidelity)
    print('Accuracy:', exp_accuracy)

    if print_excel_results:
        print('\nPrinting results')
        directory = 'results/' + dataset_name + '/' + split_name + '/'
        printer.print_characterictics_of_network(directory, data, hidden_nodes, output_condition, weights)
        print('\nPrinted chars of Network')
        printer.print_activation_values(directory, data)
        print('Printed activation values')
        printer.print_evaluation(directory, data, output_condition, bio=bio, BNN=BNN)
        print('Printed evaluation')
        printer.print_symbol_dict(data, output_condition, directory, BNN=BNN, bio=bio)
        print('Printed symbol dictionary')
        print('Finished')

    return exp_accuracy[1], fidelity[1], complexity, bio


def convert_rule(class_rule, concept_names, n_features):
    if concept_names is None:
        concept_names = [f"feature{i:010}" for i in range(n_features)]

    if isinstance(class_rule, bool):
        print("Deep Red failed in extracting rules")
        return f"{class_rule}"

    final_rule = "("
    for bio in class_rule:
        for rule in bio:
            if rule[3]:
                partial_rule = f"{concept_names[rule[1]]} > {rule[2]}"
            else:
                partial_rule = f"{concept_names[rule[1]]} < {rule[2]}"
            final_rule += partial_rule + " & "
        final_rule = final_rule[:-3]
        final_rule += ") | ("
    return final_rule[:-4]


def convert_rules(rules, concept_names, n_features):
    final_rules = []
    for class_rule in rules:
        rule = convert_rule(class_rule, concept_names, n_features)
        final_rules.append(rule)
    return final_rules


def train_deepred(dataset_name, n_features, hidden_nodes, classes,
                  trainindx=None, testindx=None, model_name='novel_model_2', seed=0, concept_names=None):

    model_name += "_" + dataset_name
    split_name = str(seed)
    if trainindx is None or testindx is None:
        set_split(dataset_name, split_name, 70)
        trainindx, testindx = lr.load_indexes(dataset_name, split_name)
    else:
        trainindx = np.random.choice(trainindx, size=len(trainindx)//100, replace=False)
        # testindx = np.random.choice(testindx, size=10000, replace=False)
        set_split_manually(dataset_name, split_name, train_indexes=trainindx, test_indexes=testindx)
    print("Train idx len:", len(trainindx))
    print("Test idx len:", len(testindx))
    # set_cv_folds(dataset_name, 3)

    # if not os.path.isfile(os.path.join("models", model_name + ".ckpt.index")):
    train_acc, test_acc = prepare_network(dataset_name, split_name, model_name, hidden_nodes,
                                          init_iterations=1000, wsp_iterations=100, wsp_accuracy_decrease=0.02,
                                          rxren_accuracy_decrease=5, function='tanh', softmax=True)

    exp_accuracies, fidelities, complexities, rules = [], [], [], []
    for target_class, _ in enumerate(classes):
        old_stdout = sys.stdout
        sys.stdout = None
        t = time.time()
        exp_accuracy, fidelity, complexity, bio = extract_model(dataset_name, split_name, model_name,
                                                                hidden_nodes, target_class)
        sys.stdout = old_stdout
        print(f"{target_class}/{len(classes)} rules extracted. Time: {time.time() - t}")
        exp_accuracies.append(exp_accuracy)
        fidelities.append(fidelity)
        complexities.append(complexity)
        rules.append(bio)

    print("Rules extracted. Starting converting...")
    rules = convert_rules(rules, concept_names, n_features)
    for i, rule in enumerate(rules):
        print(f"{i}/{len(rules)} rule: {rule}")

    return test_acc*100, exp_accuracies*100, fidelities*100, complexities, rules


if __name__ == "__main__":
    # trainindx = list(range(64, 683))
    # testindx = list(range(64))
    train_deepred(dataset_name='breast-cancer-wisconsinBinary', n_features=90, hidden_nodes=[30, 16, 2], classes=[1])
