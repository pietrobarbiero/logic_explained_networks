# Builds a binary decision tree

import sys
import operator
import math
from . import simplification as s

class decisionnode:
	def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
		'''
		Initializes a tree node
		'''
		self.col=col
		self.value=value
		self.results=results
		self.tb=tb
		self.fb=fb

def divideset(rows, column, value):
	'''
	Divides the dataset according to an attribute.

    param row -- row in dataset
    param column -- column in dataset
    param value -- dividing threshold

	'''
	set1 = []
	set2 = []
	for row in rows:
		if row[column]>value:
			set1.append(row)
		else:
			set2.append(row)
	return (set1,set2)

def uniquecounts(rows):
	'''
	Returns the belonging of the instances to the classes

    param rows -- the dataset in the current branch, the class is the argument
    in the last column

	'''
	results={}
	for row in rows:
		r=row[len(row)-1]
		if r not in results: results[r]=0
		results[r]+=1
	return results

def giniimpurity(rows):
	'''
	Returns the gini impurity (the probability that an item is 
	wrongly classified)

    param rows -- the dataset in the current branch

	'''
	total=len(rows)
	counts=uniquecounts(rows)
	imp=0
	for k1 in counts:
		p1=float(counts[k1])/total
		for k2 in counts:
			if k1==k2: continue
			p2=float(counts[k2])/total
			imp+=p1*p2
	return imp

def entropy(rows):
	'''
	Calculates the Entropy = sum of p(x)log(p(x)) across all 
	the different possible results

    param rows -- the dataset in the current branch

	'''
	from math import log
	log2=lambda x:log(x)/log(2)
	results=uniquecounts(rows)
	ent=0.0
	for r in results.keys():
		p=float(results[r])/len(rows)
		ent=ent-p*log2(p)
	return ent

def get_dnfs(cond_layer, tree):
	'''
	Returns a dnf belonging to class '0' and one for belonging to class '1',
	where each conjunction is a set of conditions of the type 
	(layer, neuron, threshold, bigger?)

	param cond_layer -- the layer the condition referrs to
	param tree -- the decision tree
	'''
	dnf = [None] * 2
	dnf[0] = []
	dnf[1] = []
	def return_rules(cond_layer, tree, conditions):
		if tree.results!=None:
			tree_class = max(tree.results.items(), key=operator.itemgetter(1))[0] #ADDED iteritems -> items in Python3
			simplified_rule = s.delete_redundant_terms(conditions)
			if simplified_rule:
				simplified_rule.sort()
				if simplified_rule not in dnf[int(tree_class)]:
					dnf[int(tree_class)].append(simplified_rule)
		else:
			condition_f = (cond_layer, tree.col, tree.value, False)
			if condition_f not in conditions:
				return_rules(cond_layer, tree.fb, conditions + [condition_f])
			else:
				return_rules(cond_layer, tree.fb, conditions)
			condition_t = (cond_layer, tree.col, tree.value, True)
			if condition_t not in conditions:
				return_rules(cond_layer, tree.tb, conditions + [condition_t])
			else:
				return_rules(cond_layer, tree.tb, conditions)
	return_rules(cond_layer, tree, [])
	if len(dnf[0]) == 0:
		dnf[0] = False
		dnf[1] = True
	elif len(dnf[1]) == 0:
		dnf[1] = False
		dnf[0] = True
	return dnf

def getwidth(tree):
	'''
	Returns the width of the tree

    param tree -- the decision tree

	'''
	if tree.tb==None and tree.fb==None: return 1
	return getwidth(tree.tb)+getwidth(tree.fb)

def getdepth(tree):
	'''
	Returns the depth of the tree

    param tree -- the decision tree

	'''
	if tree.tb==None and tree.fb==None: return 0
	return max(getdepth(tree.tb),getdepth(tree.fb))+1

def variance(rows):
	'''
	Calculates the class variance of the remaining data

    param rows -- the dataset in the current branch

	'''
	if len(rows)==0: return 0
	data=[float(row[len(row)-1]) for row in rows]
	mean=sum(data)/len(data)
	variance=sum([(d-mean)**2 for d in data])/len(data)
	return variance

def buildtree(rows, split_points, scoref=entropy, class_dominance = 98, min_set_size = 1, max_depth = 30, root = False):
  '''
  Builds a decision tree in a recursive manner

  param rows -- the dataset in the current branch
  param split_points -- for each column, the threshold that could be used to divide that column
  param scoref -- the measure used to assess the pureness of the data
  param root -- only used at the begining if the tree is at its root
  Stopping criteria:
  param class_dominance: a percentage applied to the current database size. 
  If that number of examples are classified correctly without further 
  increasing the tree, it stops growing, calculated on each run
  param min_set_size -- a fixed number previously calulated using the size of
  the initial dataset. If the current dataset is smaller than that number, 
  the tree stops growing
  param param max_depth: is a set number outlying the maximal depth of the tree
  '''
  print('buildtree')
  #print('rows',rows)
  if len(rows)==0: return decisionnode()
  current_classification = uniquecounts(rows)
  print('current_classification', current_classification)
  # If more than this number of examples belongs to the predicted class,
  # the tree is not further split
  for_class_dominance = (float(len(rows))*float(class_dominance))/100.0
  examples_mayority_class = max(current_classification.values())
  # If it is not the first split and one of the criteria has been reached
  if not root and (len(rows) <= min_set_size or max_depth == 0 or examples_mayority_class >= for_class_dominance):
	  if len(rows) <= min_set_size: print('Return for lack of examples')
	  if max_depth == 0: print('Return for reaching max depth')
	  if examples_mayority_class >= for_class_dominance: print('Return for class dominance')
	  print('unique counts', current_classification)
	  return decisionnode(results=current_classification)
  current_score=scoref(rows)
  #best criteria
  best_gain=-1
  best_criteria=None
  best_sets=None
  column_count=len(rows[0])-1
  for col in range(0,column_count):
    column_values={}
    for sp in split_points[col]:
       column_values[sp]=1
    # Tries to divide the attribute by each split point
    for value in column_values.keys():
      (set1,set2)=divideset(rows,col,value)
      # Information gain
      p=float(len(set1))/len(rows)
      gain=current_score-p*scoref(set1)-(1-p)*scoref(set2)
      if gain>=best_gain and len(set1)>0 and len(set2)>0:
        best_gain=gain
        best_criteria=(col,value)
        best_sets=(set1,set2)

  #sub tree
  if (best_criteria and best_sets and len(best_sets[0])>min_set_size and len(best_sets[1])>min_set_size) or root:
  	trueBranch=buildtree(best_sets[0], split_points, scoref, class_dominance, min_set_size, max_depth-1, root = False)
  	falseBranch=buildtree(best_sets[1], split_points, scoref, class_dominance, min_set_size, max_depth-1, root = False)
  	return decisionnode(col=best_criteria[0],value=best_criteria[1],tb=trueBranch,fb=falseBranch)
  else:
    return decisionnode(results=current_classification)

def decimals(x):
    max_digits = 14
    int_part = int(abs(x))
    magnitude = 1 if int_part == 0 else int(math.log10(int_part)) + 1
    if magnitude >= max_digits:
        return 0
    frac_part = abs(x) - int_part
    multiplier = 10 ** (max_digits - magnitude)
    frac_digits = multiplier + int(multiplier * frac_part + 0.5)
    while frac_digits % 10 == 0:
        frac_digits /= 10
    return int(math.log10(frac_digits))

