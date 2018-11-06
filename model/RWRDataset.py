import torch
import torch.utils.data as data

import sys

if sys.version_info > (3, 4):
	import _pickle as pickle
else:
	import pickle as pickle



def load_pickle(fin):

	with open(fin, 'rb') as f:
		obj = pickle.load(f)
	return obj


class RWRDataset(data.Dataset):

	def __init__(self, dataset_split_file):

		self.dataset_split_file = dataset_split_file
		self.dataset = []
		for sample in load_pickle(dataset_split_file):
			self.dataset.append( sample )


	def __getitem__(self, index):

		return self.dataset[index]


	def __len__(self):

		return len(self.dataset)
