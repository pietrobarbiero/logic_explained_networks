# Performs post-pruning of a rule set by deleting rules that don't improve the training accuracy

import simplification as s
import evaluation_formulas as ef
from operator import concat
import time
import math
import itertools
import heapq

def post_prune(dnf, class_condition, condition_example_dict, example_indexes, data = None):
	'''
	Applies boolean simplification and post-pruning
	
	param dnf: list of lists emulating a DNF expression
	param class_condition: condition which that DNF classifies for
	param condition_example_dict: evaluation of conditions in the network
	param example_indexes: training examples
	'''
	if len(dnf) > 1:
		dnf = delete_non_ocurring_rules(dnf, class_condition, condition_example_dict, example_indexes)
		for r in dnf:
			r.sort()
		dnf, positives, negatives, tp, fp = _build_pos_neg_tp_fp(dnf, class_condition, condition_example_dict, example_indexes)
		#start = time.time()
		dnf = prune_rules(dnf, class_condition, condition_example_dict, example_indexes, positives, negatives, tp, fp)
		print('\nRules that do not increase accuracy are pruned')
		#end = time.time()
		print('F number rules:',len(dnf))
		print('F number terms:',sum(len(r) for r in dnf))
		if data:
			print('Fidelity:', ef.accuracy_of_dnf(data, class_condition, dnf, True, False, False, True))
		#print('TIME: ', end-start)
	if len(dnf) > 1:
		dnf = s.boolean_simplify_complex(dnf)
		print('\nBasic boolean simplification')
		#end = time.time()
		print('F number rules:',len(dnf))
		print('F number terms:',sum(len(r) for r in dnf))
		if data:
			print('Fidelity:', ef.accuracy_of_dnf(data, class_condition, dnf, True, False, False, True))
		#print('TIME: ', end-start)
	return dnf

def delete_non_ocurring_rules(dnf, class_condition, condition_example_dict, example_indexes):
	positives = [e for e in example_indexes if _fulfills_condition(e, condition_example_dict, class_condition)]
	n_rules = len(dnf)
	return [dnf[r] for r in range(n_rules) if any(_fulfills_rule(e, condition_example_dict, dnf[r]) for e in positives)]

def _create_merged_rule(rule_1, rule_2):
	""" Finds a rule which is the least general rule that is more general than
	both argument rules.
	"""
	neurons = set((l, n) for (l, n, t, b) in rule_1).intersection((l, n) for (l, n, t, b) in rule_1)
	new_rule = []
	for (l_i, n_i) in neurons:
		bigger_t_1 = [t for (l, n, t, b) in rule_1 if (l, n) == (l_i, n_i) and b]
		smaller_t_1 = [t for (l, n, t, b) in rule_1 if (l, n) == (l_i, n_i) and not b]
		bigger_t_2 = [t for (l, n, t, b) in rule_2 if (l, n) == (l_i, n_i) and b]
		smaller_t_2 = [t for (l, n, t, b) in rule_2 if (l, n) == (l_i, n_i) and not b]
		if bigger_t_1 and bigger_t_2:
			min_t = min(bigger_t_1 + bigger_t_2)
			new_rule.append((l_i, n_i, min_t, True))
		if smaller_t_1 and smaller_t_2:
			max_t = max(smaller_t_1 + smaller_t_2)
			new_rule.append((l_i, n_i, max_t, False))
	return new_rule

def _fulfills_condition(example_index, condition_example_dict, cond):
	if cond[3]:
		if example_index in condition_example_dict[(cond[0], cond[1], cond[2])]:
			return True
		else:
			return False
	else:
		if example_index not in condition_example_dict[(cond[0], cond[1], cond[2])]:
			return True
		else:
			return False
			
def _fulfills_rule(example_index, condition_example_dict, rule):
	if all(_fulfills_condition(example_index, condition_example_dict, c) for c in rule):
		return True
	else:
		return False

def _build_pos_neg_tp_fp(dnf, class_condition, condition_example_dict, example_indexes, positives=None, negatives=None):
	if not positives:
		positives = [e for e in example_indexes if _fulfills_condition(e, condition_example_dict, class_condition)]
	if not negatives:
		negatives = [e for e in example_indexes if not _fulfills_condition(e, condition_example_dict, class_condition)]
	tp = []
	fp = []
	remaining_rules = []
	for r in range(len(dnf)):
		tp_r = [e for e in positives if _fulfills_rule(e, condition_example_dict, dnf[r])]
		fp_r = [e for e in negatives if _fulfills_rule(e, condition_example_dict, dnf[r])]
		if len(fp_r) < len(tp_r):
			tp.append(tp_r)
			fp.append(fp_r)
			remaining_rules.append(r)
	dnf = [dnf[r] for r in remaining_rules]
	return dnf, positives, negatives, tp, fp

def prune_rules(dnf, class_condition, condition_example_dict, example_indexes, positives, negatives, tp, fp):
	'''
	The rules are ordered according to their increasing certainty factor. Starting from the first rule, 
	the change of fidelity of deleting it, merging it with any other rule or of eliminating any of its conditions is calculated. 
	The change that brings the best improvement is considered. If this improvement is cero or more, the change is made, 
	the fidelity is updated and the modified rule is inserted into the heap of remaining rules. The other side of the 
	rule remains in the heap, but would probably be deleted at the end.
	'''
	n_rules = len(dnf)
	# This stores the indexes of the rules that will be kept at the end
	remaining_rules = range(n_rules)
	# This stores the indexes of the rules that are yet to be seen. They are ordered in terms of increasing certainty factor.
	rules_to_explore = []
	for r in range(n_rules):
		try:
			certainty_factor = float(len(tp[r]))/(len(tp[r]) + len(fp[r]))
		except ZeroDivisionError:
			certainty_factor = 0
		heapq.heappush(rules_to_explore, (certainty_factor, r))
	tp_count = sum(1 for e in positives if any(e in tp[i] for i in remaining_rules))
	tn_count = sum(1 for e in negatives if not any(e in fp[i] for i in remaining_rules))
	last_accuracy = float(tp_count+tn_count)/len(example_indexes)
	print('last_accuracy', last_accuracy)
	while rules_to_explore:
		print('rules_to_explore', rules_to_explore)
		print('Size of heap:', len(rules_to_explore))
		cf, j = heapq.heappop(rules_to_explore)
		print('j', j)
		#print(dnf[j])
		# Positive and negative examples that are classified by the rest of the rules
		tp_others = [e for e in positives if any(e in tp[i] for i in remaining_rules if i != j)]
		fp_others = [e for e in negatives if any(e in fp[i] for i in remaining_rules if i != j)]
		# Positive and negative examples that are not classified by any of the other rules
		left_tp = [e for e in positives if e not in tp_others]
		left_fp = [e for e in negatives if e not in fp_others]
		# Positive and negative examples that are only classified by the rule
		tp_rule = [e for e in tp[j] if e not in tp_others]
		fp_rule = [e for e in fp[j] if e not in fp_others]
		rule_variants = []
		# If each of its conditions is deleted
		for cond_i in range(len(dnf[j])):
			new_rule = dnf[j][:]
			del new_rule[cond_i]
			additional_tp = [e for e in left_tp if _fulfills_rule(e, condition_example_dict, new_rule)]
			additional_fp = [e for e in left_fp if _fulfills_rule(e, condition_example_dict, new_rule)]
			accuracy = float(len(tp_others)+len(additional_tp)+len(left_fp)-len(additional_fp))/len(example_indexes)
			new_rule.sort()
			heapq.heappush(rule_variants, (-accuracy, new_rule))
		# Possible merging with any other rule
		for other in remaining_rules:
			if other != j:
				new_rule = _create_merged_rule(dnf[j], dnf[other])
				new_rule.sort()
				if new_rule != sorted(dnf[j]) and new_rule != sorted(dnf[other]):
					additional_tp = [e for e in left_tp if _fulfills_rule(e, condition_example_dict, new_rule)]
					additional_fp = [e for e in left_fp if _fulfills_rule(e, condition_example_dict, new_rule)]
					accuracy = float(len(tp_others)+len(additional_tp)+len(left_fp)-len(additional_fp))/len(example_indexes)					
					heapq.heappush(rule_variants, (-accuracy, new_rule))
		# If the rule is deleted
		accuracy = float(len(tp_others)+len(left_fp))/len(example_indexes)
		heapq.heappush(rule_variants, (-accuracy, []))
		# Get the change that leads to the smallest decrease in accuracy
		accuracy, chosen_rule = heapq.heappop(rule_variants)
		if -accuracy > last_accuracy:
			last_accuracy = -accuracy
			print('last_accuracy', last_accuracy)
			# The rule is deleted
			remaining_rules.remove(j)
			# If the chosen rule is not the empty rule, it is added to the heap, and it is added to the dnf at the end.
			# Also, its true positive and true negative examples are added to the respective arrays
			print('new rule', chosen_rule)
			if len(chosen_rule)>0:
				tp_new = [e for e in positives if _fulfills_rule(e, condition_example_dict, chosen_rule)]
				fp_new = [e for e in negatives if _fulfills_rule(e, condition_example_dict, chosen_rule)]
				tp.append(tp_new)
				fp.append(fp_new)
				dnf.append(chosen_rule)
				remaining_rules.append(n_rules)
				try:
					certainty_factor = float(len(tp_new))/(len(tp_new) + len(fp_new))
				except ZeroDivisionError:
					certainty_factor = 0
				heapq.heappush(rules_to_explore, (certainty_factor, n_rules))
				n_rules += 1
			else:
				print('rule deleted')
	return [dnf[r] for r in remaining_rules]
