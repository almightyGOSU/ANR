import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from FILEPATHS import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn.modules.module import _addindent
from torch.autograd import Variable

from .RWRDataset import RWRDataset

import numpy as np

if sys.version_info > (3, 4):
	import _pickle as pickle
else:
	import pickle as pickle



# Tokens - NOTE: In the vocabulary, <pad> has index 0, and <unk> has index 1
PAD_TAG = "<pad>"
UNK_TAG = "<unk>"

PAD_idx = 0
UNK_idx = 1


# For formatting
TEXT_SEP = "========================================================================================================================"


def select_gpu(gpu):

	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
	print("\n[utilities.py\\select_gpu] os.environ[\"CUDA_VISIBLE_DEVICES\"]: {}".format( os.environ["CUDA_VISIBLE_DEVICES"] ))


# This is for PyTorch 0.3.1 and below
# Variables & Tensors are merged in PyTorch 0.4.0, refer to https://pytorch.org/2018/04/22/0_4_0-migration-guide.html
def to_var(x, use_cuda = False, phase = "Train"):

	# phase = {Train, Dev, Test}
	if(use_cuda):
		x = x.cuda()
	return Variable(x, volatile = (False if phase == "Train" else True))


# https://stackoverflow.com/a/45528544/4112664
def torch_summarize(model, show_weights = True, show_parameters = True, show_trainable = True):

	"""Summarizes torch model by showing trainable parameters and weights."""
	tmpstr = model.__class__.__name__ + " (\n"

	for key, module in model._modules.items():

		# If it contains layers let call it recursively to get params and weights
		if type(module) in [
			torch.nn.modules.container.Container,
			torch.nn.modules.container.Sequential
		]:
			modstr = torch_summarize(module)
		else:
			modstr = module.__repr__()

		# ====================== Extra stuff (for displaying nn.Parameter) ======================
		lst_params = []
		for name, p in module.named_parameters():
			if(type(p) == torch.nn.parameter.Parameter and "weight" not in name and "bias" not in name):
				lst_params.append("  ({}): Parameter{}".format( name, tuple(p.size()) ))

		if(lst_params):
			modstr = modstr[:-1]
			modstr += "\n".join(lst_params)
			modstr += "\n)"
		# ====================== Extra stuff (for displaying nn.Parameter) ======================

		modstr = _addindent(modstr, 2)

		weights = tuple([tuple(p.size()) for p in module.parameters()])
		params = sum([np.prod(p.size()) for p in module.parameters()])

		total_params = sum([torch.LongTensor(list(p.size())).prod() for p in module.parameters()])
		trainable_params = sum([torch.LongTensor(list(p.size())).prod() for p in module.parameters() if p.requires_grad])

		tmpstr += "  (" + key + "): " + modstr 
		if show_weights:
			tmpstr += ", weights = {}".format(weights)
		if show_parameters:
			tmpstr +=  ", parameters = {:,}".format(params)
		if show_trainable and total_params != 0 and total_params == trainable_params:
			tmpstr +=  " (Trainable)"
		tmpstr += "\n"   

	tmpstr = tmpstr + ")"
	return tmpstr


# Generates a summary of the given model
def generate_mdl_summary(mdl, logger):

	model_size = sum([np.prod(p.size()) for p in mdl.parameters()])
	logger.log("\nModel Size: {:,}".format( model_size ))

	trainable_model_size = sum([ np.prod(p.size()) for p in filter(lambda p: p.requires_grad, mdl.parameters()) ])
	logger.log("# of Trainable Parameters: {:,}".format( trainable_model_size ))

	logger.log(torch_summarize(mdl))
	logger.log(TEXT_SEP, print_txt = False)


# https://www.python.org/dev/peps/pep-0485/
# PEP 485 -- A Function for testing approximate equality
def isclose(a, b, rel_tol = 1e-09, abs_tol = 0.0):

	return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


# Creating a folder
def mkdir_p(path):

	if path == "":
		return
	try:
		os.makedirs(path)
	except:
		pass


def load_pickle(fin):

	with open(fin, 'rb') as f:
		obj = pickle.load(f)
	return obj


# Returns the # of users, # of items, and the average train rating
def loadInfo(args):

	info_path = "{}{}{}".format( args.input_dir, args.dataset, fp_info )
	print("\nLoading 'info' from \"{}\"..".format( info_path ))
	info = load_pickle(info_path)
	print("'info' loaded!")
	return info['num_users'], info['num_items']


# Loads the training, validation, and testing sets
def loadTrainDevTest(logger, args):

	train_path 	= "{}{}{}".format( args.input_dir, args.dataset, fp_split_train )
	dev_path 	= "{}{}{}".format( args.input_dir, args.dataset, fp_split_dev )
	test_path 	= "{}{}{}".format( args.input_dir, args.dataset, fp_split_test )

	print("\nLoading training set from \"{}\"..".format( train_path ))
	train_set = RWRDataset(train_path)
	train_loader = data.DataLoader(dataset = train_set, batch_size = args.batch_size, shuffle = True, num_workers = 0)
	print("Training set loaded! Note: Training examples are shuffled every epoch, i.e. shuffle = True!")

	print("\nLoading validation set from \"{}\"..".format( dev_path ))
	dev_set = RWRDataset(dev_path)
	dev_loader = data.DataLoader(dataset = dev_set, batch_size = args.batch_size, shuffle = False, num_workers = 0)
	print("Validation set loaded!")

	print("\nLoading testing set from \"{}\"..".format( test_path ))
	test_set = RWRDataset(test_path)
	test_loader = data.DataLoader(dataset = test_set, batch_size = args.batch_size, shuffle = False, num_workers = 0)
	print("Testing set loaded!")

	logger.log("\nTrain/Dev/Test splits loaded! |TRAIN|: {:,}, |DEV|: {:,}, |TEST|: {:,}".format( len(train_set), len(dev_set), len(test_set) ))
	return train_set, train_loader, dev_set, dev_loader, test_set, test_loader


# Best Dev MSE & the corresponding Test MSE/MAE
def getBestPerf(lstDevMSE, lstTestMSE, lstTestMAE):

	lstDevTest = []
	for devMSE, testMSE, testMAE in zip(lstDevMSE, lstTestMSE, lstTestMAE):
		lstDevTest.append( [devMSE, testMSE, testMAE] )

	lstDevTest = sorted(lstDevTest, key = lambda item: item[0])

	# Best Dev MSE & the corresponding Test MSE/MAE
	bestDevMSE = lstDevTest[0][0]
	testMSE_forBestDevMSE = lstDevTest[0][1]
	testMAE_forBestDevMSE = lstDevTest[0][2]

	# Epoch number
	epoch_num_forBestDevMSE = lstDevMSE.index(bestDevMSE) + 1

	return epoch_num_forBestDevMSE, bestDevMSE, testMSE_forBestDevMSE, testMAE_forBestDevMSE


