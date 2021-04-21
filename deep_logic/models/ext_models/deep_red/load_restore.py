import pickle
import os.path
import numpy as np

def get_metadata(experiment_id):
	a = experiment_id.split('_')
	dataset_name = a[0]
	class_spec = dataset_name[-4:]
	if '-C' in class_spec:
		target_class_index = int(class_spec[class_spec.find('C')+1:])
		dataset_whole = dataset_name[:dataset_name.find('-C')]
	else:
		target_class_index = 1
		dataset_whole = dataset_name
	model_name = None
	aa = a[1].split('-')
	if aa[0] == 's':
		model_name = 'models/'+dataset_name+'_shallow'
	#elif aa[0] == 'deep':
	#	model_name = 'models/'+dataset_name+'_deep'
	#elif aa[0] == 'manual':
	#	model_name = 'models/'+dataset_name+'_manual'
	else:
		print('Model not identified')
		exit()
	if len(aa) == 1:
		model_name = model_name + '.ckpt'
	elif len(aa) == 3:
		model_name = model_name + '_polarized_pruned.ckpt'
	elif len(aa) == 2:
		if aa[1] == 'pol':
			model_name = model_name + '_polarized.ckpt'
		elif aa[1] == 'pr':
			model_name = model_name + '_pruned.ckpt'
		else:
			print('Model not identified')
			exit()
	else:
		print('Model not identified')
		exit()
	hidden_nodes, function, softmax, beta = get_model_metadata(model_name)
	# Split is a porcentage or a string or the form fln-m or fhn-m
	split = a[2]
	# train_folds has elements of the form 'cvn-m'
	train, test, train_folds = load_indexes(split, dataset_name)
	if 'fl' in split and (split[2:] != train_folds[0][2:] or len(train_folds)>1):
		print('Error, folds dont match')
		exit()
	if 'tv1' in a[3]:
		vali = None
	else:
		vali = load_vali_split(dataset_whole, train_folds, a[3])
	aa = a[4].split(',')
	class_dominance = int(aa[0])
	min_set_size = int(aa[1])
	aa = a[5].split(',')
	dis_config = int(aa[0])
	rft_pruning_config = int(aa[1])
	rep_pruning_config = int(aa[2])
	return dataset_name, model_name, hidden_nodes, softmax, train_folds, train, test, vali, class_dominance, min_set_size, dis_config, rft_pruning_config, rep_pruning_config, target_class_index

def save_cluster_means(means, experiment, k='100', split = '100'):
	a = experiment.split('_')
	class_spec = a[0][-4:]
	if '-C' in class_spec:
		a[0] = a[0][:a[0].find('-C')]
	name = a[0]+'_'+a[1]+'_'+str(k)+'_'+split+'.pkl'
	with open('cluster_means/'+name, 'wb') as f:
		pickle.dump(means, f, pickle.HIGHEST_PROTOCOL)

def save_cut_points(cut_points, experiment, k='100', split = '100'):
	a = experiment.split('_')
	class_spec = a[0][-4:]
	if '-C' in class_spec:
		a[0] = a[0][:a[0].find('-C')]
	name = a[0]+'_'+a[1]+'_'+str(k)+'_'+split+'.pkl'
	with open('cut_points/'+name, 'wb') as f:
		pickle.dump(cut_points, f, pickle.HIGHEST_PROTOCOL)

def load_cluster_means(experiment, k='100', split = '100'):
	a = experiment.split('_')
	class_spec = a[0][-4:]
	if '-C' in class_spec:
		a[0] = a[0][:a[0].find('-C')]
	name = a[0]+'_'+a[1]+'_'+str(k)+'_'+split+'.pkl'
	with open('cluster_means/'+name, 'rb') as f:
		return pickle.load(f)

def load_cut_points(experiment, k='100', split = '100'):
	a = experiment.split('_')
	class_spec = a[0][-4:]
	if '-C' in class_spec:
		a[0] = a[0][:a[0].find('-C')]
	name = a[0]+'_'+a[1]+'_'+str(k)+'_'+split+'.pkl'
	with open('cut_points/'+name, 'rb') as f:
		return pickle.load(f)

def save_vali_split(vali_indexes, name, dataset, split):
	with open('indexes/'+dataset+'_'+str(split)+'_'+name+'.pkl', 'wb') as f:
		pickle.dump(vali_indexes, f, pickle.HIGHEST_PROTOCOL)

def load_vali_split(dataset, train_folds, name):
	vali = []
	for f in train_folds:
		with open('indexes/'+dataset+'_'+f+'_'+name+'.pkl', 'rb') as f:
			vali.extend(pickle.load(f))
	return vali


def load_indexes(dataset_name, split_name):
	if 'cv' in split_name:
		train = []
		test = None
		side = split_name[:2]
		split = split_name[2:].split('-')
		folds = [f for f in range(int(split[0])) if not f == int(split[1])]
		train_splits = ['cv'+split[0]+'-'+str(f) for f in folds]
		for ts in train_splits:
			with open('indexes/'+dataset_name+'_'+ts+'.pkl', 'rb') as f:
				train.extend(pickle.load(f))
		with open('indexes/'+dataset_name+'_'+split_name+'.pkl', 'rb') as f:
			test = pickle.load(f)
	else:
		with open('indexes/'+dataset_name+'_'+split_name+'_train.pkl', 'rb') as f:
			train = pickle.load(f)
		with open('indexes/'+dataset_name+'_'+split_name+'_test.pkl', 'rb') as f:
			test = pickle.load(f)
	return train, test
		
def save_train_indexes(train_indexes, dataset_name, split_name):
	with open('indexes/'+dataset_name+'_'+split_name+'_train.pkl', 'wb') as f:
		pickle.dump(train_indexes, f, pickle.HIGHEST_PROTOCOL)

def save_test_indexes(test_indexes, dataset_name, split_name):
	with open('indexes/'+dataset_name+'_'+split_name+'_test.pkl', 'wb') as f:
		pickle.dump(test_indexes, f, pickle.HIGHEST_PROTOCOL)

def save_indexes(indexes, dataset_name, split_name):
	with open('indexes/'+dataset_name+'_'+split_name+'.pkl', 'wb') as f:
		pickle.dump(indexes, f, pickle.HIGHEST_PROTOCOL)

def save_BNN_ecd_indexes(BNN, example_cond_dict, index_list, name):
	# Last letter refers to the post-pruning configuration, which does not
	# affect any of these objects
	with open('obj/BNN_'+ name + '.pkl', 'wb') as f:
		pickle.dump(BNN, f, pickle.HIGHEST_PROTOCOL)
	with open('obj/example_cond_dict_'+ name + '.pkl', 'wb') as f:
		pickle.dump(example_cond_dict, f, pickle.HIGHEST_PROTOCOL)
	with open('obj/index_list_'+ name + '.pkl', 'wb') as f:
		pickle.dump(index_list, f, pickle.HIGHEST_PROTOCOL)

def save_bio(bio, name):
	with open('obj/bio_'+ name + '.pkl', 'wb') as f:
		pickle.dump(bio, f, pickle.HIGHEST_PROTOCOL)

def save_baseline(baseline, name):
	with open('obj/baseline_'+ name + '.pkl', 'wb') as f:
		pickle.dump(baseline, f, pickle.HIGHEST_PROTOCOL)

def save_c45(baseline, name):
	with open('obj/c45_'+ name + '.pkl', 'wb') as f:
		pickle.dump(baseline, f, pickle.HIGHEST_PROTOCOL)

def save_act_values_paramaters(act_train, act_test, weights, bias, model_name, split):
	name = model_name[7:-5]
	with open('nn_data/act_train_'+ name + '_' + str(split) + '.pkl', 'wb') as f:
		pickle.dump(act_train, f, pickle.HIGHEST_PROTOCOL)
	with open('nn_data/act_test_'+ name + '.pkl', 'wb') as f:
		pickle.dump(act_test, f, pickle.HIGHEST_PROTOCOL)
	with open('nn_data/weights_'+ name + '.pkl', 'wb') as f:
		pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)
	with open('nn_data/bias_'+ name + '.pkl', 'wb') as f:
		pickle.dump(bias, f, pickle.HIGHEST_PROTOCOL)

def load_BNN_ecd_indexes(name):
	result = []
	with open('obj/BNN_' + name + '.pkl', 'rb') as f:
		result.append(pickle.load(f))
	with open('obj/example_cond_dict_'+ name + '.pkl', 'rb') as f:
		result.append(pickle.load(f))
	with open('obj/index_list_'+ name + '.pkl', 'rb') as f:
		result.append(pickle.load(f))
	return result

def load_act_values_paramaters(model_name, train_folds):
	name = model_name[7:-5]
	result = []
	train_activations = None
	for fold in train_folds:
		with open('nn_data/act_train_'+ name + '_' + fold + '.pkl', 'rb') as f:
			if not train_activations:
				train_activations = pickle.load(f)
			else:
				new_activations = pickle.load(f)
				for layer in range(len(train_activations)):
					train_activations[layer] = np.append(train_activations[layer], new_activations[layer], axis=0)
	result.append(train_activations)
	with open('nn_data/act_test_'+ name + '.pkl', 'rb') as f:
		result.append(pickle.load(f))
	with open('nn_data/weights_'+ name + '.pkl', 'rb') as f:
		result.append(pickle.load(f))
	with open('nn_data/bias_'+ name + '.pkl', 'rb') as f:
		result.append(pickle.load(f))
	return result

def load_bio(name):
	with open('obj/bio_' + name + '.pkl', 'rb') as f:
		return pickle.load(f)

def load_baseline(name):
	with open('obj/baseline_' + name + '.pkl', 'rb') as f:
		return pickle.load(f)

def load_c45(name):
	with open('obj/c45_' + name + '.pkl', 'rb') as f:
		return pickle.load(f)
