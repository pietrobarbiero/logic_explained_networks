# Replaces conditions of some layer with those of a shallower layer to 
# create the expression of the output activations with conditions that 
# refer only to the dataset arguments

from . import simplification as s
from . import pruning as p
from . import evaluation_formulas as ef
from operator import concat
import itertools
#from sympy.logic.boolalg import is_dnf
import time
from sympy import *
import sys
from functools import reduce

def handle_boolean_values(deep_rule, d):
	'''
	Replaces list with boolean values when necessary
	
	The deep rule is made up of several conditions, each which the dictionary
	mirrors to a rule with shallower conditions. If all conditions are mirrored to
	a 'True' rule, then True should be returned. If only some are mirrored to 'True',
	these should be deleted. If there is at least one condition mirrored to a 'False'
	value, then 'False' should be returned.
	'''
	remain = []
	for c in range(len(deep_rule)):
		mirror = d[deep_rule[c]]
		if not isinstance(mirror, list):
			if mirror == False:
				return False
		else:
			remain.append(c)
	rule = [deep_rule[i] for i in remain]
	if len(rule) > 0:
		return rule
	else:
		return True


def replace_rules(DNF, d):
	'''
	Replaces the conditions of the current expression with conditions
	of the next shallower layer
	
    param DNF -- DNF at layer
    param d -- BNN
	'''
	if not isinstance(DNF, list):
			return DNF
	rules = set([])
	num_cs = len(DNF)
	i = 0
	for c in DNF:		
		i+=1
		print('Conjunction '+str(i)+' from '+str(num_cs))
		h = handle_boolean_values(c, d)
		if isinstance(h, list):
			for x in itertools.product(*(d[i] for i in h)):
				rule = s.delete_redundant_terms(reduce(concat, x))
				if rule:
					s.insert_non_redundant(rules, rule)
		else:
			if h == True:
				return True
	if len(rules) > 0:
		return [list(t) for t in rules]
	else:
		return False

def get_bio(BNN, output_condition, example_cond_dict, dict_indexes, with_data = 1, data = None):
	'''
	Gets expression of the outputs using input values, which can be interpreted as a 
	rule set which simils the bahaviour of the network without showing how it works
	
    param output_condition -- condition of interest
    param example_cond_dict -- used for post-pruning, shows how the network stands to a 
    param condition according to an example
    param dict_indexes -- determined the indexes which should be considered for post-pruning
    param with_data -- 2 means use of post-pruning is activated, 1 that it is not. Avoid using 0.

	'''
	print('\nGetting bio')
	f = BNN[output_condition]
	condition_layer = output_condition[0]-2
	while condition_layer >= 0:
		print('')
		print('Condition layer', condition_layer)
		start = time.time()
		f = replace_rules(f, BNN)
		if not isinstance(f, list):
			return f
		
		print('\nReplaced terms')
		#print('F number rules:',len(f))
		#print('F number terms:',sum(len(r) for r in f))
		end = time.time()
		print('TIME: ', end-start)
		if data:
			print('Fidelity:', ef.accuracy_of_dnf(data, output_condition, f, True, False, False, True))
		if with_data == 0:
			f = s.boolean_simplify_basic(f)
		elif with_data >= 1:
			start = time.time()
			f = s.boolean_simplify_complex(f)
			end = time.time()
			print('TIME:'+str(end-start))
			#print('\nBasic boolean simplification')
			#print('F number rules:',len(f))
			#print('F number terms:',sum(len(r) for r in f))
			#print('TIME: ', end-start)
		if with_data == 2 and len(f)>1:
			#print('TIME: ', end-start)
			#start = time.time()
			f = p.post_prune(f, output_condition, example_cond_dict, dict_indexes, data=data)
			#end = time.time()
		condition_layer -= 1
	return f



