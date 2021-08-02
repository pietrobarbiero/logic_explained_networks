import xlsxwriter
import pickle
from . import evaluation_formulas as ef
import os
import time

def print_summary(experiments, bio_info, baseline_info, C45_info, BNN_info, vali, dataset = None):
	vali = False
	row = 0
	directory = 'results/'
	if dataset:
		file_name = directory + '/evaluation_summary_'+dataset+'_'+time.strftime("%d-%H-%M")+'.xlsx'
	else:
		file_name = directory + '/evaluation_summary_'+time.strftime("%d-%m-%H-%M-%S")+'.xlsx'
	workbook = xlsxwriter.Workbook(file_name)
	worksheet = workbook.add_worksheet()
	bold = workbook.add_format({'bold': True})	
	worksheet.set_landscape()
	worksheet.set_margins(left=0.3, right=0.3, top=0.3, bottom=0.3)
	r = 0
	worksheet.write(row, r, 'Dataset', bold)
	r = r+1
	worksheet.write(row, r, 'Model', bold)
	r = r+1
	worksheet.write(row, r, 'Split', bold)
	r = r+1
	worksheet.write(row, r, 'Train-validation split', bold)
	r = r+1
	worksheet.write(row, r, 'C4.5 parameters', bold)
	r = r+1
	worksheet.write(row, r, 'Discretization', bold)
	r = r+1
	worksheet.write(row, r, 'From trees pruning', bold)
	r = r+1
	worksheet.write(row, r, 'Replacement pruning', bold)
	r = r+1
	worksheet.write(row, r, 'F # conditions', bold)
	r = r+1
	worksheet.write(row, r, 'F # rules', bold)
	r = r+1
	worksheet.write(row, r, 'F # distinct split points', bold)
	#r = r+1
	#worksheet.write(row, r, 'F train fidelity to original', bold)
	#r = r+1
	#if vali:
	#	worksheet.write(row, r, 'F vali fidelity to original', bold)
	#	r = r+1
	#worksheet.write(row, r, 'F test fidelity to original', bold)
	r = r+1
	worksheet.write(row, r, 'F train fidelity', bold)
	r = r+1
	worksheet.write(row, r, 'F test fidelity', bold)
	r = r+1
	worksheet.write(row, r, 'F train precision to nn', bold)
	r = r+1
	worksheet.write(row, r, 'F test precision to nn', bold)
	r = r+1
	worksheet.write(row, r, 'F train recall to nn', bold)
	r = r+1
	worksheet.write(row, r, 'F test recall to nn', bold)
	r = r+1
	worksheet.write(row, r, 'F train accuracy', bold)
	r = r+1
	if vali:
		worksheet.write(row, r, 'F vali accuracy', bold)
		r = r+1
	worksheet.write(row, r, 'F test accuracy', bold)
	r = r+1
	worksheet.write(row, r, '# Intermediate entries', bold)
	r = r+1
	worksheet.write(row, r, '# Intermediate conditions', bold)
	r = r+1
	worksheet.write(row, r, '# Intermediate rules', bold)
	r = r+1
	worksheet.write(row, r, '# Intermediate distinct split points', bold)
	r = r+1
	worksheet.write(row, r, '# Intermediate average thresholds per used neuron', bold)
	r = r+1
	worksheet.write(row, r, '# Conditions per layer, starting with shallower', bold)
	r = r+1
	worksheet.write(row, r, 'Average train fidelity per layer', bold)
	r = r+1
	worksheet.write(row, r, 'Average test fidelity per layer', bold)
	r = r+1
	worksheet.write(row, r, 'Baseline # conditions', bold)
	r = r+1
	worksheet.write(row, r, 'Baseline # rules', bold)
	r = r+1
	worksheet.write(row, r, 'Baseline # distinct split points', bold)	
	r = r+1
	#worksheet.write(row, r, 'Baseline train fidelity to original', bold)
	#r = r+1
	#if vali:
	#	worksheet.write(row, r, 'Baseline vali fidelity to original', bold)
	#	r = r+1
	#worksheet.write(row, r, 'Baseline test fidelity to original', bold)
	#r = r+1
	worksheet.write(row, r, 'Baseline train fidelity', bold)
	r = r+1
	worksheet.write(row, r, 'Baseline test fidelity', bold)
	r = r+1
	worksheet.write(row, r, 'Baseline train precision to nn', bold)
	r = r+1
	worksheet.write(row, r, 'Baseline test precision to nn', bold)
	r = r+1
	worksheet.write(row, r, 'Baseline train recall to nn', bold)
	r = r+1
	worksheet.write(row, r, 'Baseline test recall to nn', bold)
	r = r+1
	worksheet.write(row, r, 'Baseline train accuracy', bold)
	r = r+1
	if vali:
		worksheet.write(row, r, 'Baseline vali accuracy', bold)
		r = r+1
	worksheet.write(row, r, 'Baseline test accuracy', bold)
	r = r+1
	worksheet.write(row, r, 'C4.5 # conditions', bold)
	r = r+1
	worksheet.write(row, r, 'C4.5 # rules', bold)
	r = r+1
	worksheet.write(row, r, 'C4.5 # distinct split points', bold)
	#r = r+1
	#worksheet.write(row, r, 'C4.5 train fidelity to original', bold)
	#r = r+1
	#if vali:
	#	worksheet.write(row, r, 'C4.5 vali fidelity to original', bold)
	#	r = r+1
	#worksheet.write(row, r, 'C4.5 test fidelity to original', bold)
	r = r+1
	worksheet.write(row, r, 'C4.5 train fidelity', bold)
	r = r+1
	worksheet.write(row, r, 'C4.5 test fidelity', bold)
	r = r+1
	worksheet.write(row, r, 'C4.5 train precision to nn', bold)
	r = r+1
	worksheet.write(row, r, 'C4.5 test precision to nn', bold)
	r = r+1
	worksheet.write(row, r, 'C4.5 train recall to nn', bold)
	r = r+1
	worksheet.write(row, r, 'C4.5 test recall to nn', bold)
	r = r+1
	worksheet.write(row, r, 'C4.5 train accuracy', bold)
	r = r+1
	if vali:
		worksheet.write(row, r, 'C4.5 vali accuracy', bold)
		r = r+1
	worksheet.write(row, r, 'C4.5 test accuracy', bold)

	def dnf_info(row, info, col, f):
		formats[1].set_fg_color('#C6FFB3')
		formats[2].set_fg_color('#FFFFCC')
		formats[3].set_fg_color('#FFE5CC')
		formats[4].set_fg_color('#EFCCFF')
		formats[5].set_fg_color('#E6F2FF')
		worksheet.write(row, col, info[0], formats[1])
		worksheet.write(row, col+1, info[1], formats[2])
		worksheet.write(row, col+2, info[2], formats[3])
		if vali:
			# Fidelity
			worksheet.write(row, col+7, info[5][2], formats[4])
			# Accuracy
			worksheet.write(row, col+13, info[4][1], formats[5])
			worksheet.write(row, col+14, info[4][2], formats[5])
			# Fidelity to original network
			worksheet.write(row, col+3, info[3][0], formats[0])
			worksheet.write(row, col+4, info[3][1], formats[4])
			worksheet.write(row, col+5, info[3][2], formats[4])
			worksheet.write(row, col+6, info[5][0], formats[0])
			worksheet.write(row, col+8, info[6][0], formats[0])
			worksheet.write(row, col+9, info[6][2], formats[0])
			worksheet.write(row, col+10, info[7][0], formats[0])
			worksheet.write(row, col+11, info[7][2], formats[0])
			worksheet.write(row, col+12, info[4][0], formats[0])
		else:
			# Fidelity
			worksheet.write(row, col+4, info[4][1], formats[4])
			# Accuracy
			worksheet.write(row, col+10, info[3][1], formats[5])
			# Fidelity to original network
			#worksheet.write(row, col+3, info[3][0], formats[0])
			#worksheet.write(row, col+4, info[3][1], formats[4])
			worksheet.write(row, col+3, info[4][0], formats[0])
			worksheet.write(row, col+5, info[5][0], formats[0])
			worksheet.write(row, col+6, info[5][1], formats[0])
			worksheet.write(row, col+7, info[6][0], formats[0])
			worksheet.write(row, col+8, info[6][1], formats[0])
			worksheet.write(row, col+9, info[3][0], formats[0])
		
	def BNN_print(row, info, col, f):
		worksheet.write(row, col, info[0], f)
		worksheet.write(row, col+1, info[1], f)
		worksheet.write(row, col+2, info[2], f)
		worksheet.write(row, col+3, info[3], f)
		worksheet.write(row, col+4, info[4], f)
		worksheet.write(row, col+5, str(info[5]), f)
		worksheet.write(row, col+6, str(info[6]), f)
		worksheet.write(row, col+7, str(info[7]), f)		
	for e in range(len(experiments)):
		row += 1
		col = 0
		exp = experiments[e].split('_')
		config = exp[5].split(',')
		formats = [None] * 6
		for f in range(6):
			formats[f] = workbook.add_format()		
			if config[1] == '2' and config[2] == '1':
				formats[f].set_italic()
			if config[0] == '0' or config[0] == '5':
				formats[f].set_font_color('#006600')
			elif config[0] == '1' or config[0] == '3':
				formats[f].set_font_color('#5D0885')
			elif config[0] == '2' or config[0] == '4':
				formats[f].set_font_color('#003399')
		worksheet.write(row, col, exp[0], formats[0])
		worksheet.write(row, col+1, exp[1], formats[0])
		worksheet.write(row, col+2, exp[2], formats[0])
		worksheet.write(row, col+3, exp[3], formats[0])
		worksheet.write(row, col+4, exp[4], formats[0])
		worksheet.write(row, col+5, config[0], formats[0])
		worksheet.write(row, col+6, config[1], formats[0])
		worksheet.write(row, col+7, config[2], formats[0])
		col += 8
		dnf_info(row, bio_info[e], col, formats)
		if vali:
			col += 15
		else:
			col += 11
		BNN_print(row, BNN_info[e], col, formats[0])
		col += 8
		dnf_info(row, baseline_info[e], col, formats)
		if vali:
			col += 15
		else:
			col += 11
		dnf_info(row, C45_info[e], col, formats)
	workbook.close()

def print_activation_values(directory, data):
	if not os.path.exists(directory):
		os.makedirs(directory)
	file_name = directory + '/activation_values.xlsx'
	workbook = xlsxwriter.Workbook(file_name)
	worksheet = workbook.add_worksheet()
	bold = workbook.add_format({'bold': True})
	worksheet.write(0, 0, 'EXAMPLE', bold)
	row = 1
	col = 0
	for index in data.train_indexes + data.vali_indexes + data.test_indexes:
		x_format = workbook.add_format()
		if index in data.train_indexes:
			x_format.set_bg_color('#E4EEF6')
		if index in data.vali_indexes:
			x_format.set_bg_color('#FADCE1')
		elif index in data.test_indexes:
			x_format.set_bg_color('#DEC1E6')
		worksheet.write(row, 0, index)
		observation = data.examples[index]
		values = observation.values
		col = 1
		for layer in range(len(values)):
			for node in range(len(values[layer])):
				worksheet.write(0, col, 'h'+str(layer)+'_'+str(node), bold)
				worksheet.write(row, col, values[layer][node], x_format)
				col += 1
		worksheet.write(0, col, 'CLASS', bold)
		worksheet.write(row, col, observation.class_value, x_format)
		row += 1
	workbook.close()
	
def print_symbol_dict(data, output_condition, root_directory, BNN = None, BNNi = None, bio = None, baseline = None):
	directory = root_directory	
	if not os.path.exists(directory):
		os.makedirs(directory)
	file_name = directory + 'symbol_dict.xlsx'
	# Create a workbook and add a worksheet.
	workbook = xlsxwriter.Workbook(file_name)
	worksheet = workbook.add_worksheet()
	# Start from the first cell. Rows and columns are zero indexed.
	row = 0
	# First column is for the example index
	example_column = 0
	worksheet.write(row, example_column, 'EXAMPLE')
	keys = set([])
	if BNN:
		keys.update(BNN.keys());
	if BNNi:
		keys.update(BNNi.keys());
	keys = list(keys)
	keys.sort()
	bold = workbook.add_format({'bold': True})
	# This dictionary links each condition with an index for the column
	symbol_columns = {}
	col = 1
	# Inputs
	for i in range(data.input_lenght):
		symbol_columns[(0, i)] = col
		worksheet.write(row, col, str((0, i)))
		col += 1
	for k in keys:
		symbol_columns[k] = col
		worksheet.write(row, col, str(k))
		col += 1
		if BNN and (k in BNN):
			worksheet.write(row, col, 'BNN '+str(k)+': '+str(BNN[k]))
			col += 1
		if BNNi and (k in BNN):
			worksheet.write(row, col, 'BNNi '+str(k)+': '+str(BNN[k]))
			col += 1
	if bio:
		worksheet.write(row, col, 'bio')
		bio_col = col
		col += 1
	if baseline:
		worksheet.write(row, col, 'baseline')
		baseline_col = col
		col += 1
	train_indexes = data.train_indexes
	test_indexes = data.test_indexes
	row = 1
	def get_format(index, match=-1):
		curr_format = workbook.add_format()
		if index in train_indexes:
			curr_format.set_bg_color('#E4EEF6')
		elif index in test_indexes:
			curr_format.set_bg_color('#DEC1E6')
		if match >= 0:
			if match == 0:
				curr_format.set_font_color('red')
				curr_format.set_bold()
			else:
				curr_format.set_font_color('green')
		return curr_format
	examples = data.examples
	for e in train_indexes + test_indexes:
		current_format = get_format(e)
		worksheet.write(row, example_column, e, current_format)
		for k, c in symbol_columns.items(): #ADDED iteritems -> items in Python3
			if len(k) == 2:
				current_format = get_format(e)
				worksheet.write(row, c, str(examples[e].values[0][k[1]]), current_format)
			else:
				current_format = get_format(e)
				v = examples[e].fulfills_cond(k)
				worksheet.write(row, c, v, current_format)
				if BNN and k in BNN:
					f = examples[e].fulfills_dnf(BNN[k])
					if v == f:
						match = 1
					else:
						match = 0
					current_format = get_format(e, match)
					worksheet.write(row,c+1, f, current_format)
				if BNNi and k in BNNi:
					f = examples[e].fulfills_dnf(BNNi[k])
					if v == f:
						match = 1
					else:
						match = 0
					current_format = get_format(e, match)
					worksheet.write(row,c+2, f, current_format)
		v = examples[e].fulfills_cond(output_condition)
		if bio:
			f = examples[e].fulfills_dnf(bio)
			if v == f:
				match = 1
			else:
				match = 0
			current_format = get_format(e, match)
			worksheet.write(row, bio_col, f, current_format)
		if baseline:
			f = examples[e].fulfills_dnf(baseline)
			if v == f:
				match = 1
			else:
				match = 0
			current_format = get_format(e, match)
			worksheet.write(row, baseline_col, f, current_format)	
		row +=1
	workbook.close()
	
def print_characterictics_of_network(directory, data, hidden_nodes, output_condition, weights, col_start = 0, row_start = 0):
	col = col_start
	row = row_start
	
	output_char = (output_condition[1], output_condition[2])
	
	if not os.path.exists(directory):
		os.makedirs(directory)
	file_name = directory + '/chars_of_network.xlsx'
	workbook = xlsxwriter.Workbook(file_name)
	worksheet = workbook.add_worksheet()
	worksheet.write(row, col, 'CHARACTERISTICS OF THE TRAINED NETWORK')
	row += 2
	worksheet.write(row, col, 'HIDDEN NODES: ' + str(hidden_nodes))
	row += 2
	worksheet.write(row, col, 'TRAINING CHARACTERISTICS')
	row += 1
	worksheet.write(row, col, 'Average standard deviation from center:')
	avg_deviation = ef.avg_neuron_deviation_from_center(data, hidden_nodes)
	worksheet.write(row, col+1, str(avg_deviation))
	row += 1
	worksheet.write(row, col, 'Porcentage zero weights:')
	zero_weights = ef.porcentace_zero_weights(weights)
	worksheet.write(row, col+1, str(zero_weights)+'%')
	row += 1
	worksheet.write(row, col, 'Porcentage zero activations:')
	zero_acts = ef.porcentage_zero_activations(data, hidden_nodes)
	worksheet.write(row, col+1, str(zero_acts)+'%')
	row += 2
	worksheet.write(row, col, 'SEMANTIC CHARACTERISTICS')
	row += 1
	vali = data.num_vali > 0
	test = data.num_test > 0
	
	accuracy = ef.network_accuracy(output_char, data)
	precision = ef.network_precision(output_char, data)
	recall = ef.network_recall(output_char, data)
	worksheet.write(row, col, 'Accuracy:')
	worksheet.write(row+1, col, 'Precision:')
	worksheet.write(row+2, col, 'Recall:')
	worksheet.write(row-1, col+1, 'Train')
	worksheet.write(row, col+1, str(accuracy[0]))
	worksheet.write(row+1, col+1, str(precision[0]))
	worksheet.write(row+2, col+1, str(recall[0]))
	if vali:
		worksheet.write(row-1, col+2, 'Validation')
		worksheet.write(row, col+2, str(accuracy[1]))
		worksheet.write(row+1, col+2, str(precision[1]))
		worksheet.write(row+2, col+2, str(recall[1]))
	if test:
		worksheet.write(row-1, col+3, 'Test')
		worksheet.write(row, col+3, str(accuracy[2]))
		worksheet.write(row+1, col+3, str(precision[2]))
		worksheet.write(row+2, col+3, str(recall[2]))
	workbook.close()

def print_evaluation(root_directory, data, output_condition, bio=None, baseline=None, BNN=None, BNNi=None, col_start = 0, row_start = 0, per_BNN_entry = False):
	directory = root_directory
	if not os.path.exists(directory):
		os.makedirs(directory)
	r = row_start
	c = col_start
	file_name = directory + 'evaluation.xlsx'
	workbook = xlsxwriter.Workbook(file_name)
	worksheet = workbook.add_worksheet()
	bold = workbook.add_format({'bold': True})
	vali = data.num_vali > 0
	test = data.num_test > 0
	tr_i = 0
	if test:
		test_i = 1
	if vali:
		tv_i = 0
		tr_i = 1
		v_i = 2
		if test:
			test_i = 3
	def print_dnf_info(dnf, row, col):
		worksheet.write(row, col, '# CONDITIONS:')
		worksheet.write(row, col+1, ef.number_conditions(dnf))
		worksheet.write(row+1, col, '# RULES:')
		worksheet.write(row+1, col+1, ef.number_rules(dnf))
		worksheet.write(row+2, col, '# DISTINCT SPLIT POINTS:')
		worksheet.write(row+2, col+1, ef.num_distinct_split_points(dnf))
	def print_BNN_info(BNN, row, col):
		worksheet.write(row, col, '# ENTRIES:')
		worksheet.write(row, col+1, ef.number_entries(BNN))
		worksheet.write(row+1, col, '# CONDITIONS:')
		worksheet.write(row+1, col+1, ef.BNN_number_conditions(BNN))
		worksheet.write(row+2, col, '# RULES:')
		worksheet.write(row+2, col+1, ef.BNN_number_rules(BNN))
		worksheet.write(row+3, col, '# DISTINCT SPLIT POINTS:')
		worksheet.write(row+3, col+1, ef.BNN_num_distinct_split_points(BNN))
		worksheet.write(row+4, col, '# AVERAGE THRESHOLDS PER USED NEURON:')
		worksheet.write(row+4, col+1, ef.BNN_avg_thresholds_used_neurons(BNN))
		per_layer_info = ef.per_layer_info(data, BNN, output_condition[0])
		worksheet.write(row+5, col, '# CONDITIONS PER LAYER, STARTING WITH SHALLOWER:')
		worksheet.write(row+5, col+1, str(per_layer_info[0]))
		worksheet.write(row+6, col, 'AVERAGE TRAIN FIDELITY PER LAYER, STARTING WITH SHALLOWER:')
		worksheet.write(row+6, col+1, str(per_layer_info[1]))
		worksheet.write(row+7, col, 'AVERAGE TEST FIDELITY PER LAYER, STARTING WITH SHALLOWER:')
		worksheet.write(row+7, col+1, str(per_layer_info[2]))
		
	def print_score_table(dnf, key, row, col):
		row += 1
		a_i = 1
		f_i = 2
		p_i = 3
		r_i = 4
		m_i = 5
		worksheet.write(row+a_i, col, 'ACCURACY:')
		worksheet.write(row+f_i, col, 'FIDELITY:')
		worksheet.write(row+p_i, col, 'FIDELITY PRECISION:')
		worksheet.write(row+r_i, col, 'FIDELITY RECALL:')
		worksheet.write(row+m_i, col, 'MISCLASSIFIED INDEXES:')
		A = ef.class_accuracy(data, dnf, output_condition[1], t_v = vali, tr = True, v = vali, te = test)
		F = ef.accuracy_of_dnf(data, key, dnf, t_v = vali, tr = True, v = vali, te = test)
		P = ef.precision_of_dnf(data, key, dnf, t_v = vali, tr = True, v = vali, te = test)
		R = ef.recall_of_dnf(data, key, dnf, t_v = vali, tr = True, v = vali, te = test)
		M = ef.example_indexes(data, key, dnf, t_v = vali, tr = True, v = vali, te = test)
		worksheet.write(row, col+1+tr_i, 'TRAIN')
		worksheet.write(row+a_i, col+1+tr_i, str(A[tr_i]))
		worksheet.write(row+f_i, col+1+tr_i, str(F[tr_i]))
		worksheet.write(row+p_i, col+1+tr_i, str(P[tr_i]))
		worksheet.write(row+r_i, col+1+tr_i, str(R[tr_i]))
		worksheet.write(row+m_i, col+1+tr_i, str(M[tr_i]))
		if vali:
			worksheet.write(row, col+1+tv_i, 'TRAIN_VAL')
			worksheet.write(row+a_i, col+1+tv_i, str(A[tv_i]))
			worksheet.write(row+f_i, col+1+tv_i, str(F[tv_i]))
			worksheet.write(row+p_i, col+1+tv_i, str(P[tv_i]))
			worksheet.write(row+r_i, col+1+tv_i, str(R[tv_i]))
			worksheet.write(row+m_i, col+1+tv_i, str(M[tv_i]))
			worksheet.write(row, col+1+v_i, 'VALIDATION')
			worksheet.write(row+a_i, col+1+v_i, str(A[v_i]))
			worksheet.write(row+f_i, col+1+v_i, str(F[v_i]))
			worksheet.write(row+p_i, col+1+v_i, str(P[v_i]))
			worksheet.write(row+r_i, col+1+v_i, str(R[v_i]))
			worksheet.write(row+m_i, col+1+v_i, str(M[v_i]))
		if test:
			worksheet.write(row, col+1+test_i, 'TEST')
			worksheet.write(row+a_i, col+1+test_i, str(A[test_i]))
			worksheet.write(row+f_i, col+1+test_i, str(F[test_i]))
			worksheet.write(row+p_i, col+1+test_i, str(P[test_i]))
			worksheet.write(row+r_i, col+1+test_i, str(R[test_i]))
			worksheet.write(row+m_i, col+1+test_i, str(M[test_i]))
			
	if bio:
		worksheet.write(r, c, 'BIO', bold)
		print_dnf_info(bio, r+1, c)
		print_score_table(bio, output_condition, r+3, c)
	if baseline:
		worksheet.write(r, c+6, 'BASELINE', bold)
		print_dnf_info(baseline, r+1, c+6)
		print_score_table(baseline, output_condition, r+3, c+6)
	if bio or baseline:
		r += 12
	if BNN:
		worksheet.write(r, c, 'BNN', bold)
		r += 1
		print_BNN_info(BNN, r, c)
		if per_BNN_entry:
			r += 8
			keys = BNN.keys()
			keys.sort(reverse=True)
			layer = keys[0][0]
			row = r
			col = c
			for key in keys:
				entry = BNN[key]
				if key[0] == layer:
					worksheet.write(r+1, col, str(key)+': '+str(entry))
					print_score_table(entry, key, r+1, col)
					col = col + 5
				else:
					layer -= 1
					r += 8
					col = c
					worksheet.write(r+1, col, str(key)+': '+str(entry))
					print_score_table(entry, key, r+1, col)
					col = col + 5
	
	workbook.close()

def avg_list(nested_list):
	new_list = []
	num_instances = len(nested_list)
	def average(elements):
		if isinstance(elements[0], list):
			return [average([elements[j][i] for j in range(len(elements))]) for i in range(max(len(elements[i]) for i in range(len(elements))))]
		else:
			return float(sum(elements))/num_instances
	for i in range(len(nested_list[0])):
		elements = [l[i] for l in nested_list]
		new_list.append(average(elements))
	return new_list

def print_cv_evaluation(directory, vali, bio_info, baseline_info, BNN_info, k, col_start = 0, row_start = 0):
	if not os.path.exists(directory):
		os.makedirs(directory)
	r = row_start
	c = col_start
	file_name = directory + 'cross_validation_'+str(k)+'_evaluation.xlsx'
	workbook = xlsxwriter.Workbook(file_name)
	worksheet = workbook.add_worksheet()
	bold = workbook.add_format({'bold': True})
	tr_i = 0
	test_i = 1
	if vali:
		tv_i = 0
		tr_i = 1
		v_i = 2
		test_i = 3
	bio_info = avg_list(bio_info)
	BNN_info = avg_list(BNN_info)
	baseline_info = avg_list(baseline_info)
	def print_dnf_info(dnf_info, row, col):
		worksheet.write(row, col, '# CONDITIONS:')
		worksheet.write(row, col+1, dnf_info[0])
		worksheet.write(row+1, col, '# RULES:')
		worksheet.write(row+1, col+1, dnf_info[1])
		worksheet.write(row+2, col, '# DISTINCT SPLIT POINTS:')
		worksheet.write(row+2, col+1, dnf_info[2])
		A = dnf_info[3]
		F = dnf_info[4]
		P = dnf_info[5]
		R = dnf_info[6]
		a_i = 1
		f_i = 2
		p_i = 3
		r_i = 4
		row += 4
		worksheet.write(row+a_i, col, 'ACCURACY:')
		worksheet.write(row+f_i, col, 'FIDELITY:')
		worksheet.write(row+p_i, col, 'FIDELITY PRECISION:')
		worksheet.write(row+r_i, col, 'FIDELITY RECALL:')
		worksheet.write(row, col+1+tr_i, 'TRAIN')
		worksheet.write(row+a_i, col+1+tr_i, str(A[tr_i]))
		worksheet.write(row+f_i, col+1+tr_i, str(F[tr_i]))
		worksheet.write(row+p_i, col+1+tr_i, str(P[tr_i]))
		worksheet.write(row+r_i, col+1+tr_i, str(R[tr_i]))
		worksheet.write(row, col+1+test_i, 'TEST')
		worksheet.write(row+a_i, col+1+test_i, str(A[test_i]))
		worksheet.write(row+f_i, col+1+test_i, str(F[test_i]))
		worksheet.write(row+p_i, col+1+test_i, str(P[test_i]))
		worksheet.write(row+r_i, col+1+test_i, str(R[test_i]))
		if vali:
			worksheet.write(row, col+1+tv_i, 'TRAIN_VAL')
			worksheet.write(row+a_i, col+1+tv_i, str(A[tv_i]))
			worksheet.write(row+f_i, col+1+tv_i, str(F[tv_i]))
			worksheet.write(row+p_i, col+1+tv_i, str(P[tv_i]))
			worksheet.write(row+r_i, col+1+tv_i, str(R[tv_i]))
			worksheet.write(row, col+1+v_i, 'VALIDATION')
			worksheet.write(row+a_i, col+1+v_i, str(A[v_i]))
			worksheet.write(row+f_i, col+1+v_i, str(F[v_i]))
			worksheet.write(row+p_i, col+1+v_i, str(P[v_i]))
			worksheet.write(row+r_i, col+1+v_i, str(R[v_i]))
	
	worksheet.write(r, c, 'BIO', bold)
	print_dnf_info(bio_info, r+1, c)
	worksheet.write(r, c+6, 'BASELINE', bold)
	print_dnf_info(baseline_info, r+1, c+6)
	r += 12
	worksheet.write(r, c, 'BNN', bold)
	r += 1
	row = r
	col = c
	worksheet.write(row, col, '# ENTRIES:')
	worksheet.write(row, col+1, BNN_info[0])
	worksheet.write(row+1, col, '# CONDITIONS:')
	worksheet.write(row+1, col+1, BNN_info[1])
	worksheet.write(row+2, col, '# RULES:')
	worksheet.write(row+2, col+1, BNN_info[2])
	worksheet.write(row+3, col, '# DISTINCT SPLIT POINTS:')
	worksheet.write(row+3, col+1, BNN_info[3])
	worksheet.write(row+4, col, '# AVERAGE THRESHOLDS PER USED NEURON:')
	worksheet.write(row+4, col+1, BNN_info[4])
	worksheet.write(row+5, col, '# CONDITIONS PER LAYER, STARTING WITH SHALLOWER:')
	worksheet.write(row+5, col+1, str(BNN_info[5]))
	worksheet.write(row+6, col, 'AVERAGE TRAIN FIDELITY PER LAYER, STARTING WITH SHALLOWER:')
	worksheet.write(row+6, col+1, str(BNN_info[6]))
	worksheet.write(row+7, col, 'AVERAGE TEST FIDELITY PER LAYER, STARTING WITH SHALLOWER:')
	worksheet.write(row+7, col+1, str(BNN_info[7]))
	workbook.close()
	
	
