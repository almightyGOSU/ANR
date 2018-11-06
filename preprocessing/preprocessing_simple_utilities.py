from __future__ import print_function

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from FILEPATHS import *

import sys
import os
import json
import shutil
import time
from collections import defaultdict, Counter
import codecs
from tqdm import tqdm
import re
import gc
import numpy as np
import random
import argparse

if sys.version_info > (3, 4):
	import _pickle as pickle
else:
	import pickle as pickle


# Padding and OOV
PAD = '<pad>'
UNK = '<unk>'


# https://stackoverflow.com/questions/14630288/unicodeencodeerror-charmap-codec-cant-encode-character-maps-to-undefined
# This version works for both Python 2 & 3
def uprint(*objects):

	sep = " "
	end = "\n"
	file = sys.stdout
	enc = file.encoding

	if enc == 'UTF-8':
		print(*objects, sep = sep, end = end, file = file)
	else:
		f = lambda obj: str(obj).encode(enc, errors = 'backslashreplace').decode(enc)
		print(*map(f, objects), sep = sep, end = end, file = file)


def append_to_file(path, txt, print = True):
	with codecs.open(path, 'a+', encoding = 'utf-8', errors = 'ignore') as f:
		f.write(txt + '\n')

	if(print == True):
		uprint(txt)


def print_args(args):

	args.command = ' '.join(sys.argv)
	items = vars(args)

	lst_args = []
	lst_args.append("[args from argparse.ArgumentParser().parse_args()]")
	for key in sorted(items.keys(), key = lambda s: s.lower()):

		value = items[key]
		lst_args.append("{}: {}".format( key, str(value) ))

	del args.command
	return "\n".join(lst_args)


def count(interactions, print_min = False):

	user_count = Counter()
	item_count = Counter()

	for interaction in interactions:
		user_count[interaction[0]] += 1
		item_count[interaction[1]] += 1

	if(print_min):
		print("Least # of reviews for an user: {}, Least # of reviews for an item: {}".format(
			user_count.most_common()[-1][1], item_count.most_common()[-1][1]))

	return user_count, item_count


def stack_count(interactions, user_count, item_count, print_min = False):

	for interaction in interactions:
		user_count[interaction[0]] += 1
		item_count[interaction[1]] += 1

	if(print_min):
		print("Least # of reviews for an user: {}, Least # of reviews for an item: {}".format(
			user_count.most_common()[-1][1], item_count.most_common()[-1][1]))


# Returns the set of users, and the set of items, within interactions
def get_users_items(interactions):

	users = set()
	items = set()

	for interaction in interactions:
		users.add(interaction[0])
		items.add(interaction[1])

	return users, items


def drop_if_lt(x_count, threshold):

	for x in list(x_count):
		if(x_count[x] < threshold):
			del x_count[x]


def hit(interaction, users_dict, items_dict):
	try:
		test_user = users_dict[interaction[0]]
		test_item = items_dict[interaction[1]]
		return True
	except:
		return False


# Replace with custom tokenizer and any desired preprocessing here
def simple_tokenizer(txt):

	# Convert to lowercase, remove new lines
	txt = txt.lower()
	txt = txt.replace("\r\n", " ").replace("\n", " ")

	# Remove punctuation
	txt = re.sub(r"[^\w\s]", " ", txt)

	# Tokenize
	txt =  [x for x in txt.split() if len(x) > 0]

	return txt


def doc2id(doc, word_wid):
	return [word2id(word, word_wid) for word in doc]


def word2id(word, word_wid):
	try:
		return word_wid[word]
	except:
		return word_wid[UNK]


def post_padding(doc, maxlen, pad_char = 0):
	return doc[:maxlen] + ([pad_char] * (maxlen - len(doc)) )


def prepare_set(interactions, user_uid, item_iid, uid_userDocLen, iid_itemDocLen, set_type, output_log, printToScreen = False):

	lst_uid = []
	lst_iid = []
	lst_rating = []

	for interaction in tqdm(interactions, "Preparing the {} set".format(set_type)):

		user = interaction[0]
		item = interaction[1]
		rating = interaction[2]

		# Convert user string to id, and item string to id
		uid = user_uid[user]
		iid = item_iid[item]

		lst_uid.append(uid)
		lst_iid.append(iid)
		lst_rating.append(rating)


	# Distribution of Ratings (Just fyi)
	ratings_min = np.min(lst_rating)
	ratings_max = np.max(lst_rating)
	append_to_file(output_log, "[{}] Lowest Rating: {:.2f}, Highest Rating: {:.2f}, Average Rating: {:.3f}".format(
		set_type, ratings_min, ratings_max, np.mean(lst_rating) ), print = printToScreen)
	ratingsCounter = Counter(lst_rating)
	ratingsDist = ratingsCounter.items()
	ratingsDist = sorted(ratingsDist, key = lambda interaction: interaction[0])
	ratingsDist = ", ".join(["[{:.2f}: {}]".format(r, c) for r, c in ratingsDist])
	append_to_file(output_log, "[{}] {}\n".format( set_type, ratingsDist ), print = printToScreen)


	return zip(lst_uid, lst_iid, lst_rating)


def load_pickle(fin):
	with open(fin, 'rb') as f:
		obj = pickle.load(f)
	return obj


# Includes startIndex, excludes endIndex
def createNumpyMatrix(startIndex, endIndex, mapping):

	npMatrix = []
	rows = (endIndex - startIndex)
	columns = len(mapping[startIndex])

	for idx in range(startIndex, endIndex):

		vec = mapping[idx]
		npMatrix.append(vec)

	npMatrix = np.stack(npMatrix)
	npMatrix = np.reshape(npMatrix, (rows, columns))

	return npMatrix

