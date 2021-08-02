# Performs boolean simplification of DNF expressions

import itertools
import math
import sys
import copy

def delete_redundant_terms(rule):
	'''
	Only a maximum of two conditions per neuron should remain, one 
	of the upper bound, ending with 'False', that should be the one
	of the lowest threshold; and one of the lower bound which should
	be the highest.
	If the true remaining thresholds are inconsistent, then the rule is
	inconsistent and the method returns None.
	>>> delete_redundant_terms([(3, 1, 0.2, False), (3, 1, 0.4, False), (3, 1, 0.1, True)], [])
		[(3, 1, 0.1, True), (3, 1, 0.2, False)]
	>>> delete_redundant_terms([(3, 1, 0.4, False), (3, 1, 0.3, True), (3, 1, 0.1, True)], [])
		[(3, 1, 0.3, True), (3, 1, 0.4, False)]
	>>> delete_redundant_terms([(3, 1, 0.2, False), (3, 1, 0.4, False), (3, 1, 0.3, True), (3, 1, 0.1, True)], [])
		None
	'''
	#print('del_redundant_terms')
	neurons = set([(l, n) for (l, n, t, b) in rule])
	new_rule = []
	for (l_i, n_i) in neurons:
		# If the neuron is discrete, there should only be one condition referring to the attribute taking some value.
		bigger_t = [t for (l, n, t, b) in rule if (l, n) == (l_i, n_i) and b]
		smaller_t = [t for (l, n, t, b) in rule if (l, n) == (l_i, n_i) and not b]
		if bigger_t:
			max_t = max(bigger_t)
			new_rule.append((l_i, n_i, max_t, True))
		if smaller_t:
			min_t = min(smaller_t)
			new_rule.append((l_i, n_i, min_t, False))
		if bigger_t and smaller_t and max_t >= min_t:
			return None
	return new_rule
	
def is_gen(rule_1, rule_2):
	'''
	Checks if rule_2 is a specification of rule_1
	'''
	#print('is_gen_check')
	if set(rule_1) != set(rule_2) and len(rule_1) <= len(rule_2):
		n_1 = set(((l, n, b) for (l, n, t, b) in rule_1))
		n_2 = set(((l, n, b) for (l, n, t, b) in rule_2))
		intersection = n_1.intersection(n_2)
		if len(n_1) == len(intersection):
			# It is possible that rule_2 is a specialization
			for (l_i, n_i, b_i) in intersection:
				t_1 = next(t for (l, n, t, b) in rule_1 if (l, n, b) == (l_i, n_i, b_i))
				t_2 = next(t for (l, n, t, b) in rule_2 if (l, n, b) == (l_i, n_i, b_i))
				if b_i:
					if t_1 > t_2:
						return False
				else:
					if t_1 < t_2:
						return False
			return True
	return False

def boolean_simplify_basic(rules):
	'''
	Without observing the data, it returns the rules that are not a specification 
	of another rule in the list.
	
	>>> r1 = [(1, 2, 0.5, True), (1, 2, 0.6, False), (1, 3, 0.3, True), (1, 3, 0.5, False)]
	>>> r2 = [(1, 2, 0.5, True), (1, 2, 0.6, False), (1, 3, 0.5, False), (1, 3, 0.2, True)]
	>>> r3 = [(1, 2, 0.5, True), (1, 2, 0.6, False), (1, 3, 0.5, False)]
	>>> r4 = [(1, 7, 0.6, False)]
	>>> boolean_simplify_basic([r3, r4, r2, r1])
	[[(1, 7, 0.6, False)], [(1, 2, 0.5, True), (1, 2, 0.6, False), (1, 3, 0.5, False)]]
	>>> boolean_simplify_basic([r1, r2])
	[[(1, 2, 0.5, True), (1, 2, 0.6, False), (1, 3, 0.5, False), (1, 3, 0.2, True)]]
	>>> boolean_simplify_basic([r3, r4, r2])
	[[(1, 7, 0.6, False)], [(1, 2, 0.5, True), (1, 2, 0.6, False), (1, 3, 0.5, False)]]
	'''
	#print('bool_simplify_basic')
	rules = set(tuple(conj) for conj in rules)
	return [list(conj) for conj in rules if not any(is_gen(r, conj) for r in rules)]
	

def insert_non_redundant(rules, rule):
	'''
	If a rule is not a specialization of another rule, it is inserted into 'rules'.
	'''
	rule = tuple(sorted(rule))
	if rule not in rules:
		if not any(is_gen(r, rule) for r in rules):
			rules.add(rule)
	return

def boolean_simplify_complex(rules):
	rules = set(tuple(sorted(conj)) for conj in rules)
	remaining_rules = rules.copy()
	new_rules = set([])
	def cover_all_dim(rule_1, rule_2):
		'''
		Returns the condition that must be deleted because it is irrelevant, wither from one rule or from both
		'''
		difference = set(rule_1).difference(set(rule_2))
		if len(difference) == 1:
			condition_1 = difference.pop()
			possible_complements = set(c2 for c2 in set(rule_2).difference(set(rule_1)) if c2[0] == condition_1[0] and c2[1] == condition_1[1] and not c2[3] == condition_1[3])
			if len(possible_complements) == 1:
				condition_2 = possible_complements.pop()
				if condition_1[3]:
					if condition_1[2] <= condition_2[2]:
						return condition_2
				else:
					if condition_1[2] >= condition_2[2]:
						return condition_2
		return None
	new_rules = remaining_rules.copy()
	while len(new_rules)>1:
		new_rules_old = new_rules.copy()
		new_rules = set([])
		for rule_1, rule_2 in itertools.product(remaining_rules, new_rules_old):
		# Both rules are equally long, they differ in one condition. These
		# conditions are such that if one is untrue, the other is true
			if len(rule_1) == len(rule_2):
				if cover_all_dim(rule_1, rule_2):
					remaining_rules.discard(rule_1)
					remaining_rules.discard(rule_2)				
					new_rules.add(tuple(set(rule_1).intersection(set(rule_2))))
		# Rule 2 is longer than rule 1. Rule 1 only has one condition not in rule 2
		# These conditions are such that if one is untrue, the other is true
			if len(rule_2) > len(rule_1):
				condition = cover_all_dim(rule_1, rule_2)
				if condition:
					remaining_rules.discard(rule_2)
					new_rules.add(tuple(c for c in rule_2 if c != condition))
			elif len(rule_1) > len(rule_2):
				condition = cover_all_dim(rule_2, rule_1)
				if condition:
					remaining_rules.discard(rule_1)
					new_rules.add(tuple(c for c in rule_1 if c != condition))
					#print('added 3')
		remaining_rules.update(new_rules)
	return [sorted(conj) for conj in remaining_rules if not any(is_gen(r, conj) for r in remaining_rules)]
