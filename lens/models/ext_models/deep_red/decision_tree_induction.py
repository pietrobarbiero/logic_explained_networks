# Starting from the target condition, which normally refers to network 
# outputs such as one class being chosen, rule sets are extracted using 
# conditions on the activations of the next shallower layer. The process 
# is repeated with each of these conditions, and so on until the conditions 
# are with respect to the network inputs.

from . import discretization as dis
from . import decision_tree as dt
from . import simplification as s
from . import pruning as p
from . import evaluation_formulas as ef
import numpy as np
import itertools
import time
import sys
import warnings
from scipy.cluster import vq


def build_BNN(data, output_condition, cd=98, mss=1, md=30, relevant_neuron_dictionary={},
              with_data=1, discretization=0, cluster_means=None):
    '''
	Starting from the target condition and until the conditions with respect 
	to the first hidden layer, it extracts a DNF that explains each condition
	using conditions of the next shallower layer
	
	param data: instance of DataSet
	param output_condition: condition of interest
	param cd: class dominance
	param mss: minimum dataset size
	param md: maximum tree depth
	param with_data: Avoid ==0. If == 1, the regular simplification operations are performed, if == 2, post-ppruning is performed
	param discretization: method used to determine the thresholds that split the activation range of each neuron
	'''
    BNN = {}
    deep_layer = data.output_layer
    target_class = [output_condition]
    print('deep layer: ')
    print(deep_layer)
    print('targetclass: ')
    print(target_class)
    while deep_layer > 0:
        target_split_values = set((l, n, t) for (l, n, t, u) in target_class)
        print('target_split_values: ')
        print(target_split_values)
        if not target_split_values:
            warnings.warn('Warning: no split points, returning current dictionary at layer: ' + str(deep_layer))
        print('Target split values', target_split_values)
        used_shallow_conditions = set([])
        current_data = temp_data(data, deep_layer - 1, target_class)
        if discretization == 0:
            split_points = dis.all_features_trivial_mid_points(current_data)
        elif discretization == 1 or discretization == 3:
            split_points = dis.one_time_discretization(current_data, discretization, rnd=relevant_neuron_dictionary,
                                                       tsv=list(target_split_values))
        elif discretization == 2 or discretization == 4:
            split_points = cluster_means[deep_layer - 1]
        elif discretization == 6:
            colum = [[d[c] for d in current_data] for c in range(len(current_data[0]) - 1)]
            split_points = [[sum(vq.kmeans(v, 2)[0]) / 2] for v in colum]
        elif discretization == 5:
            if deep_layer == 1:
                split_points = [[0.5] for l in range(len(current_data[0]) - 1)]
            else:
                split_points = [[0] for l in range(len(current_data[0]) - 1)]
        print('Split points', [len(l) for l in split_points])
        # print(split_points)

        print('')
        for i in target_split_values:
            print('')
            print('i: ', i)
            t = time.time()
            i_data = temp_data(data, deep_layer - 1, i)
            tree = None
            if relevant_neuron_dictionary and discretization == 0:
                pruned_split_points = [_sp(j, i, split_points, relevant_neuron_dictionary) for j in
                                       range(len(split_points))]
                # print('Pruned split points', pruned_split_points)
                tree = dt.buildtree(i_data, pruned_split_points, class_dominance=cd, min_set_size=mss, max_depth=md,
                                    root=True)
            else:
                tree = dt.buildtree(i_data, split_points, class_dominance=cd, min_set_size=mss, max_depth=md, root=True)
            if not tree:
                cero_class = sum(1 for x in i_data if x[-1] == 0)
                one_class = sum(1 for x in i_data if x[-1] == 1)
                if cero_class > one_class:
                    BNN[(i[0], i[1], i[2], True)] = False
                    BNN[(i[0], i[1], i[2], False)] = True
                else:
                    BNN[(i[0], i[1], i[2], False)] = True
                    BNN[(i[0], i[1], i[2], True)] = False
                break
            print('Tree is formed')
            print('Time: ', time.time() - t)
            dnfs = dt.get_dnfs(deep_layer - 1, tree)
            print('DNF:')
            print(dnfs)
            if (i[0], i[1], i[2], False) in target_class:
                print('False case')
                pruned = None
                if isinstance(dnfs[0], list):
                    print('Fidelity pre-pruning:', ef.accuracy_of_dnf(data, (i[0], i[1], i[2], False), dnfs[0], True, False, False, True))
                    print('Precision pre-pruning:', ef.precision_of_dnf(data, (i[0], i[1], i[2], False), dnfs[0], True, False, False, True))
                    print('Recall pre-pruning:', ef.recall_of_dnf(data, (i[0], i[1], i[2], False), dnfs[0], True, False, False, True))
                    data.update_dictionary([(l, n, t) for conj in dnfs[0] for (l, n, t, u) in conj])
                    if with_data == 0:
                        pruned = s.boolean_simplify_basic(dnfs[0])
                    elif with_data >= 1:
                        pruned = s.boolean_simplify_complex(dnfs[0])
                    if with_data == 2:
                        pruned = p.post_prune(pruned, (i[0], i[1], i[2], False), data.example_cond_dict,
                                              data.dict_indexes, data=None)
                    used_shallow_conditions.update(set(c for conj in pruned for c in conj))
                else:
                    pruned = dnfs[0]
                print('Fidelity post-pruning:', ef.accuracy_of_dnf(data, (i[0], i[1], i[2], False), pruned, True, False, False, True))
                print('Precision post-pruning:', ef.precision_of_dnf(data, (i[0], i[1], i[2], False), pruned, True, False, False, True))
                print('Recall post-pruning:', ef.recall_of_dnf(data, (i[0], i[1], i[2], False), pruned, True, False, False, True))
                BNN[(i[0], i[1], i[2], False)] = pruned
                print((i[0], i[1], i[2], False), pruned)
            if (i[0], i[1], i[2], True) in target_class:
                print('True case')
                pruned = None
                if isinstance(dnfs[1], list):
                    print('Fidelity pre-pruning:', ef.accuracy_of_dnf(data, (i[0], i[1], i[2], True), dnfs[1], True, False, False, True))
                    print('Precision pre-pruning:', ef.precision_of_dnf(data, (i[0], i[1], i[2], True), dnfs[1], True, False, False, True))
                    print('Recall pre-pruning:', ef.recall_of_dnf(data, (i[0], i[1], i[2], True), dnfs[1], True, False, False, True))
                    data.update_dictionary([(l, n, t) for conj in dnfs[1] for (l, n, t, u) in conj])
                    if with_data == 0:
                        pruned = s.boolean_simplify_basic(dnfs[1])
                    elif with_data >= 1:
                        pruned = s.boolean_simplify_complex(dnfs[1])
                    if with_data == 2:
                        pruned = p.post_prune(pruned, (i[0], i[1], i[2], True), data.example_cond_dict,
                                              data.dict_indexes, data=None)
                    used_shallow_conditions.update(set(c for conj in pruned for c in conj))
                else:
                    pruned = dnfs[1]
                print('Fidelity post-pruning:', ef.accuracy_of_dnf(data, (i[0], i[1], i[2], True), pruned, True, False, False, True))
                print('Precision post-pruning:', ef.precision_of_dnf(data, (i[0], i[1], i[2], True), pruned, True, False, False, True))
                print('Recall post-pruning:', ef.recall_of_dnf(data, (i[0], i[1], i[2], True), pruned, True, False, False, True))
                BNN[(i[0], i[1], i[2], True)] = pruned
                print((i[0], i[1], i[2], True), pruned)
        deep_layer -= 1
        target_class = list(used_shallow_conditions)
    return BNN


def target_class(class_conditions, deep_values):
    def split_point_side(sp):
        if deep_values[sp[1]] > sp[2]:
            return 1
        else:
            return 0

    if isinstance(class_conditions, tuple):
        return split_point_side(class_conditions)
    elif isinstance(class_conditions, list):
        return ''.join(str(split_point_side(sp)) for sp in class_conditions)


def temp_data(data, shallow, tc, deep=None):
    '''
	 param data: the dataset
	 type data: DataSet
	 param shallow: shallow layer index
	 type shallow: int
	 param target_class: list of split points
	 type target_class: list of (int, int, float) tuples
	 return: a dataset that includes all instances from the train and
	valdation sets made of the attributes of the shallow layer and a class
	made up of a concatenation of the target_class values
	 rtype: list of lists
	'''
    if not deep:
        deep = shallow + 1
    return [list(e.values[shallow]) + [target_class(tc, e.values[deep])]
            for e in data.get_train_obs()]


def _sp(shallow_n, deep_n, split_points, dic):
    if shallow_n in dic[(deep_n[0], deep_n[1])]:
        return split_points[shallow_n]
    else:
        return []


def relevant_neurons(weights, hidden_nodes, input_len, output_len=2):
    output_layer = len(hidden_nodes) + 1
    relevant_neurons_dictionary = {}
    # The softmax layer should not have nulled out entries
    for o in range(output_len):
        relevant_neurons_dictionary[(output_layer, o)] = range(hidden_nodes[-1])
    # Determine relevant neurons as those where the connection is not nulled
    for h in range(2, output_layer):
        deep_len = hidden_nodes[h - 1]
        shallow_len = hidden_nodes[h - 2]
        w = weights[h - 1]
        for j in range(deep_len):
            relevant_neurons_dictionary[(h, j)] = [i for i in range(shallow_len) if w.item(i, j)]
    # Connection between input and first hidden layer
    w = weights[0]
    for j in range(hidden_nodes[0]):
        relevant_neurons_dictionary[(1, j)] = [i for i in range(input_len) if w.item(i, j)]
    return relevant_neurons_dictionary
