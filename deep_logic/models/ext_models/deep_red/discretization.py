# Different methods to preslect the thresholds to divide the activation range of each neuron

from . import decision_tree as dt
import numpy as np
import heapq
import itertools
import math
import sys

sys.setrecursionlimit(10000)


# This discretization method is based on the paper "Dynamic Discretization
# of Continuous Attributes" by Gama et.al. "Dynamic" refers to the fact that
# that interdependencies between attributes are taken into account. It is 
# a supervised discretization method.

def all_features_trivial_mid_points(data, discrete_columns=[]):
    '''
	It returns the mid-points if the attribute is continuous and the values if it is discrete.
	>>> all_features_trivial_mid_points([
		[0.1, 0.5, 0.2, '00'],
		[0.2, 0.3, 0.4, '00'],
		[0.3, 0.3, 0.2, '01'],
		[0.4, 0.5, 0.4, '01']], [1])
		[[0.15, 0.25, 0.35], [0.3, 0.5], [0.3]]
	'''
    columns = len(data[0]) - 1
    mid_points = [None] * columns
    for col in range(columns):
        value_els = list(set(e[col] for e in data))
        value_els.sort()
        if col in discrete_columns:
            mid_points[col] = value_els
        else:
            mid_points[col] = [float('%.14f' % (np.mean(value_els[i:i + 2]))) for i in range(len(value_els) - 1)]
    return mid_points


def get_class_boundary_cut_points(values):
    '''
	Returns the midpoints between each succesive pair of values that are
	of a dirrerent class. Has a precision of 14 decimals.
	
	>>> get_class_boundary_cut_points([(2, '0'), (1, '0'), (0.9, '1'), (0.8, '0'), (1.2, '0'), (2, '1'), (0.8, '1')])
	[0.85, 0.95, 1.6]
	
	: param values: list of values-class tuples
	: type values: list of float-string tuples
	: return: the list of midpoints
	: rtype: list of ints
	'''

    # Sorts by default by the first element of the tuples
    original_class_length = len(values[0][1])
    sorted_els = sorted(values)
    value_els = sorted(list(set(t[0] for t in sorted_els)))
    reduced_els = [(v, ''.join(set([s for (e, s) in sorted_els if e == v]))) for v in value_els]
    value_els = [t[0] for t in reduced_els]
    mid_points = [float('%.14f' % (np.mean(value_els[i:i + 2]))) for i in
                  range(len(reduced_els) - 1) if
                  reduced_els[i][1] != reduced_els[i + 1][1] or len(reduced_els[i][1]) > original_class_length]
    return mid_points


f = get_class_boundary_cut_points([(2, '0'), (1, '0'), (0.9, '1'), (0.8, '0'), (1.2, '0'), (2, '1'), (0.8, '1')])


def get_separating_values(values):
    '''
	Instead of returning the mean, it returns a tuple with both values
	'''
    # Sorts by default by the first element of the tuples
    original_class_length = len(values[0][1])
    sorted_els = sorted(values)
    value_els = sorted(list(set(t[0] for t in sorted_els)))
    reduced_els = [(v, ''.join(set([s for (e, s) in sorted_els if e == v]))) for v in value_els]
    value_els = [t[0] for t in reduced_els]
    mid_points = [(value_els[i], value_els[i + 1]) for i in
                  range(len(reduced_els) - 1) if
                  reduced_els[i][1] != reduced_els[i + 1][1] or len(reduced_els[i][1]) > original_class_length]
    return mid_points


def all_features_cut_points(data, rnd={}, tsv=[]):
    '''
	For a given dataset, it returns the class cut points for each feature
	: param data: a dataset in which each example is a list of attributes
	followed by a class value
	: type data: a list of lists
	: return: the midpoints por each attribute index
	: rtype: list of lists of ints
	'''

    def row_split_points(row):
        if rnd and tsv:
            # If that shallow neuron is a relevant neuron for any of the target classes
            if any(row in rnd[(tsv[n][0], tsv[n][1])] for n in range(len(tsv))):
                return get_class_boundary_cut_points([(e[row], e[-1]) for e in data])
            else:
                return []
        else:
            return get_class_boundary_cut_points([(e[row], e[-1]) for e in data])

    return [row_split_points(c) for c in range(len(data[0]) - 1)]


def all_features_cut_points_one_class(data, rnd={}, tsv=None):
    '''
	For a given dataset, it returns the class cut points for each feature
	: param data: a dataset in which each example is a list of attributes
	followed by a class value, where the class value is 0 or 1
	: type data: a list of lists
	: return: the midpoints por each attribute index
	: rtype: list of lists of ints
	'''

    def row_split_points(row):
        if rnd and tsv:
            # If that shallow neuron is a relevant neuron for the target class
            if row in rnd[(tsv[0], tsv[1])]:
                return get_class_boundary_cut_points([(e[row], str(e[-1])) for e in data])
            else:
                return []
        else:
            return get_class_boundary_cut_points([(e[row], e[-1]) for e in data])

    return [row_split_points(c) for c in range(len(data[0]) - 1)]


def all_features_separating_values(data):
    def row_split_points(row):
        return get_separating_values([(e[row], e[-1]) for e in data])

    return [row_split_points(c) for c in range(len(data[0]) - 1)]


def create_children(vector, effective_vector, number_ths):
    '''
	It creates all ways a number of thresholds as in effective_vector
	can be distributed in a way that the new distribution are better and
	adhere to the max number of thresholds in 'vector'	
	'''
    n = number_ths
    k = len(effective_vector)
    # Stars and bars
    combinations = [[b - a - 1 for a, b in zip((-1,) + c, c + (n + k - 1,))] for c in
                    itertools.combinations(range(n + k - 1), k - 1)]
    potential = [c for c in combinations if np.std(c) < np.std(effective_vector)
                 and all(c[i] <= vector[i] for i in range(k))]
    potential.sort(key=lambda v: np.std(v))
    for p in potential:
        yield (p)


def create_children_fast(vector, effective_vector):
    '''
	For each index j of v for which v[j] is above the mean, a child is 
	produced that assignes to all elements where v[i]-1 < v[j] the 
	minimum of vector[i] or mean
	>>> create_children_fast([40, 12, 4, 3, 7, 8, 9], [13, 12, 2, 3, 4, 5, 6])
	[[12, 12, 3, 3, 5, 6, 7], [13, 11, 3, 3, 5, 6, 7]]
	'''
    children = []
    mean = int(np.mean(effective_vector))
    above_mean_indexes = [i for i in range(len(effective_vector)) if effective_vector[i] > mean]
    for j in above_mean_indexes:
        child = [None] * len(effective_vector)
        for i in range(len(effective_vector)):
            if i == j:
                child[i] = effective_vector[i] - 1
            elif effective_vector[i] + 1 < effective_vector[j]:
                child[i] = min(vector[i], mean)
            else:
                child[i] = effective_vector[i]
        children.append(child)
    return children


def improvement_found(processed_vs, queued_v):
    '''
	If any of the precessed vectors is an improvement on the not processed
	one, it returns True
	>>> improvement_found([[0, 1, 0, 6, 3], [0, 2, 2, 7, 3]], [0, 1, 1, 6, 3])
	return True
	'''
    return any(all(p_v[i] <= queued_v[i] for i in range(len(queued_v))) for p_v in processed_vs)


def get_threshold_dist_score(v, f_v, w, f_w, c, k):
    '''
	Returns the heuristic loss value for one set of maximal thresholds v.
	This is made out of l(v) = f(v, w) - c.g(v, w) - k.h(v, w), where
	w is the threshold number vector if no restrictions had been made
	for the threshold search and f is the missclassification rate. g is 
	the magnitude of v and h is the standard deviation (how uneven the 
	distribution is along all neurons). Both measures are divided by the
	measure obtained by w and should be low.
	g(v, w) = \frac{\sum_i{v_i}}{\sum_i{w_i}}, goes from 0 to 1
	h(v, w) = \frac{\frac{\sum_i \left | \bar{v} - v_i \right |}{i}}
				{\frac{\sum_i \left | \bar{w} - w_i \right |}{i}}
				
	'''
    g = float(sum(v)) / sum(w)
    std_w = np.std(w)
    if std_w > 0:
        h = np.std(v) / std_w
    else:
        h = 0
    # print('f_v: ', f_v)
    # print('c*g: ', c*g)
    # print('k*h: ', k*h)
    # print('score: ', str(f_v - f_w + c*g + k*h))
    return f_v - f_w + c * g + k * h


thresholds = []
branches = []
max_splits = []


def dynamic_dictretization(dataset, c=0.01, k=0.001, max_tries=5, rnd={}, tsv=[], class_dominance=100, min_set_size=0,
                           max_depth=10):
    heap = []
    split_points = all_features_cut_points(dataset, rnd, tsv)
    # print('split_points in dynamic_dictretization: ', split_points)
    max_splits = [len(sp) for sp in split_points]
    thresholds, w, f_w = dynamic_dictretization_v(dataset, split_points, cd=class_dominance, mss=min_set_size,
                                                  md=max_depth, allowed_splits=max_splits)
    heapq.heappush(heap, (get_threshold_dist_score(w, f_w, w, f_w, c, k), thresholds))
    seen_vectors = []
    seen_vectors.append(w)
    effective_vector = w
    vector = max_splits
    print('w: ', w)
    print('score: ', heap[0][0])
    number_thresholds = sum(w)
    print('Number thresholds: ', number_thresholds)
    tries = 0
    while tries < max_tries:
        if number_thresholds > 15:
            children = create_children_fast(vector, effective_vector)
            print('Creating limited children. Children: ', list(children))
        else:
            children = create_children(vector, effective_vector, number_thresholds)
            print('Creating children normally. Children: ', list(children))
        for child in children:
            if not improvement_found(seen_vectors, child):
                print('child: ', child)
                thresholds, v, miss = dynamic_dictretization_v(dataset, split_points, cd=class_dominance,
                                                               mss=min_set_size, md=max_depth, allowed_splits=child)
                print('v: ', v)
                seen_vectors.append(v)
                score = get_threshold_dist_score(v, miss, w, f_w, c, k)
                print('score: ', score)
                heapq.heappush(heap, (score, thresholds))
                if score < heap[0][0]:
                    if sum(v) < number_thresholds:
                        number_thresholds = sum(v)
                        vector = child
                        effective_vector = v
                        tries = 0
                        print('New number thresholds: ', number_thresholds)
                        break
                else:
                    tries += 1
        print('Break from while because no combination causes an improvement')
        tries = max_tries
    return heap[0][1]


def dynamic_dictretization_v(dataset, split_points, cd, mss, md, allowed_splits=[]):
    print('Starting dynamic discretization')
    global thresholds
    thresholds = [set([]) for i in range(len(dataset[0]) - 1)]
    global misses
    misses = 0
    global branches
    branches = []
    global max_splits
    max_splits = allowed_splits
    simulated_tree_builder(dataset, split_points, class_dominance=cd, min_set_size=mss, max_depth=md)
    v = [len(t) for t in thresholds]
    f_v = float(misses) / len(dataset)
    print('Leaving dynamic discretization')
    return (thresholds, v, f_v)


def simulated_tree_builder(data, split_points, class_dominance, min_set_size, max_depth, scoref=dt.entropy):
    '''
	: param data: a dataset in which each example is a list of attributes
	followed by a class value
	: type data: a list of lists
	: param split_points: links a feature name with a set of split points
	: type split_points: dictionary where the keys are in features
	: param features: a list with the name of the features
	: type features: list of numbers, where the length is equal to that of
	an example -1
	'''
    global misses
    for_class_dominance = (float(len(data)) * float(class_dominance)) / 100.0
    current_classification = dt.uniquecounts(data)
    examples_mayority_class = max(current_classification.values())
    if len(data) <= min_set_size or examples_mayority_class >= for_class_dominance or max_depth == 0:
        print('Returning')
        # global misses
        counts_per_class = dt.uniquecounts(data).values()
        misses += (sum(counts_per_class) - max(counts_per_class))
        if branches:
            # best_branch = max(branches, key=lambda k: k[1])
            # branches.remove(best_branch)
            # simulated_tree_builder(best_branch[0], split_points, class_dominance, min_set_size, best_branch[2], scoref)
            best_branch = heapq.heappop(branches)
            simulated_tree_builder(best_branch[1], split_points, class_dominance, min_set_size, best_branch[2], scoref)
            print('New branch')
        return
    current_score = scoref(data)
    best_gain = 0.0
    best_criteria = None
    best_sets = None
    # Number of features
    column_count = len(data[0]) - 1
    # First search over the thresholds that have already been selected
    for col in range(0, column_count):
        for value in thresholds[col]:
            (set1, set2) = dt.divideset(data, col, value)
            # Information gain
            p = float(len(set1)) / len(data)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if len(set1) > 0 and len(set2) > 0 and gain > best_gain:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    # Only keep searching where there are free slots
    column_indexes = [c for c in range(0, column_count) if
                      len(thresholds[c]) < max_splits[c]]  # Available split slots
    for col in column_indexes:
        for value in split_points[col]:
            (set1, set2) = dt.divideset(data, col, value)
            # Information gain
            p = float(len(set1)) / len(data)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if len(set1) > 0 and len(set2) > 0 and gain > best_gain:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    if best_gain > 0.0:
        thresholds[best_criteria[0]].add(best_criteria[1])
        # Add new set to branches
        score0 = scoref(best_sets[0])
        score1 = scoref(best_sets[1])
        if (len(best_sets[0]) > 1 and score0 > 0 and (best_sets[0], score0)
                not in branches):
            # branches.append((best_sets[0], score0, max_depth-1))
            heapq.heappush(branches, (-score0, best_sets[0], max_depth - 1))
        if (len(best_sets[1]) > 1 and score1 > 0 and (best_sets[1], score1)
                not in branches):
            heapq.heappush(branches, (-score1, best_sets[1], max_depth - 1))
        # branches.append((score1, best_sets[1], max_depth-1))
    else:  # Add minority to the missclassification counter
        # global misses
        counts_per_class = dt.uniquecounts(data).values()
        misses += (sum(counts_per_class) - max(counts_per_class))
    # Go forward on set with highest entropy
    if branches:
        # best_branch = max(branches, key=lambda k: k[1])
        # branches.remove(best_branch)
        best_branch = heapq.heappop(branches)
        simulated_tree_builder(best_branch[1], split_points, class_dominance, min_set_size, best_branch[2], scoref)


def one_time_discretization(dataset, discretization, rnd={}, tsv=[], class_dominance=100, min_set_size=0,
                            max_depth=500):
    split_points = all_features_cut_points(dataset, rnd, tsv)
    if discretization == 1:
        max_splits = [len(sp) for sp in split_points]
    elif discretization == 3:
        max_splits = [1 for sp in split_points]
    print('max_splits', max_splits)
    thresholds, w, f_w = dynamic_dictretization_v(dataset, split_points, cd=class_dominance, mss=min_set_size,
                                                  md=max_depth, allowed_splits=max_splits)
    # print('w', w)
    return thresholds
