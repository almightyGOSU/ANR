import torch
import torch.nn as nn
import torch.nn.functional as F

from .utilities import PAD_idx, UNK_idx

from .ANR_ARL import ANR_ARL
from .ANR_AIE import ANR_AIE

from .ANR_RatingPred import ANR_RatingPred
from .ANRS_RatingPred import ANRS_RatingPred

from tqdm import tqdm



'''
This is the complete Aspect-based Neural Recommender (ANR), with ARL and AIE as its main components.
'''
class ANR(nn.Module):

	def __init__(self, logger, args, num_users, num_items):

		super(ANR, self).__init__()

		self.logger = logger
		self.args = args

		self.num_users = num_users
		self.num_items = num_items


		# User Documents & Item Documents (Input)
		self.uid_userDoc = nn.Embedding(self.num_users, self.args.max_doc_len)
		self.uid_userDoc.weight.requires_grad = False

		self.iid_itemDoc = nn.Embedding(self.num_items, self.args.max_doc_len)
		self.iid_itemDoc.weight.requires_grad = False

		# Word Embeddings (Input)
		self.wid_wEmbed = nn.Embedding(self.args.vocab_size, self.args.word_embed_dim)
		self.wid_wEmbed.weight.requires_grad = False


		# Aspect Representation Learning - Single Aspect-based Attention Network (Shared between User & Item)
		self.shared_ANR_ARL = ANR_ARL(logger, args)


		# Rating Prediction - Aspect Importance Estimation + Aspect-based Rating Prediction
		if(self.args.model == "ANR"):

			# Aspect-Based Co-Attention (Parallel Co-Attention, using the Affinity Matrix as a Feature) --- Aspect Importance Estimation
			self.ANR_AIE = ANR_AIE(logger, args)

			# Aspect-Based Rating Predictor based on the estimated Aspect-Level Importance
			self.ANR_RatingPred = ANR_RatingPred(logger, args, self.num_users, self.num_items)

		# 'Simplified Model' - Basically, ARL + simplfied network (3x FCs) for rating prediction
		# The only purpose of this is to obtain the pretrained weights for ARL
		elif(self.args.model == "ANRS"):

			# Rating Prediction using the 'Simplified Model'
			self.ANRS_RatingPred = ANRS_RatingPred(logger, args)


	def forward(self, batch_uid, batch_iid, verbose = 0):

		# Input
		batch_userDoc = self.uid_userDoc(batch_uid)
		batch_itemDoc = self.iid_itemDoc(batch_iid)

		if(verbose > 0):
			tqdm.write("batch_userDoc: {}".format( batch_userDoc.size() ))
			tqdm.write("batch_itemDoc: {}".format( batch_itemDoc.size() ))

		# Embedding Layer
		batch_userDocEmbed = self.wid_wEmbed(batch_userDoc.long())
		batch_itemDocEmbed = self.wid_wEmbed(batch_itemDoc.long())

		if(verbose > 0):
			tqdm.write("batch_userDocEmbed: {}".format( batch_userDocEmbed.size() ))
			tqdm.write("batch_itemDocEmbed: {}".format( batch_itemDocEmbed.size() ))


		# ===================================================================== User Aspect-Based Representations =====================================================================
		# Aspect-based Representation Learning for User
		if(verbose > 0):
			tqdm.write("\n[Input to ARL] batch_userDocEmbed: {}".format( batch_userDocEmbed.size() ))

		userAspAttn, userAspDoc = self.shared_ANR_ARL(batch_userDocEmbed, verbose = verbose)
		if(verbose > 0):
			tqdm.write("[Output of ARL] userAspAttn: {}".format( userAspAttn.size() ))
			tqdm.write("[Output of ARL] userAspDoc:  {}".format( userAspDoc.size() ))
		# ===================================================================== User Aspect-Based Representations =====================================================================


		# ===================================================================== Item Aspect-Based Representations =====================================================================
		# Aspect-based Representation Learning for Item
		if(verbose > 0):
			tqdm.write("\n[Input to ARL] batch_itemDocEmbed: {}".format( batch_itemDocEmbed.size() ))

		itemAspAttn, itemAspDoc = self.shared_ANR_ARL(batch_itemDocEmbed, verbose = verbose)
		if(verbose > 0):
			tqdm.write("[Output of ARL] itemAspAttn: {}".format( itemAspAttn.size() ))
			tqdm.write("[Output of ARL] itemAspDoc:  {}".format( itemAspDoc.size() ))
		# ===================================================================== Item Aspect-Based Representations =====================================================================


		if(self.args.model == "ANR"):

			# Aspect-based Co-Attention --- Aspect Importance Estimation
			userCoAttn, itemCoAttn = self.ANR_AIE(userAspDoc, itemAspDoc, verbose = verbose)

			# Aspect-Based Rating Predictor based on the estimated Aspect-Level Importance
			rating_pred = self.ANR_RatingPred(userAspDoc, itemAspDoc, userCoAttn, itemCoAttn, batch_uid, batch_iid, verbose = verbose)

		# 'Simplified Model' - Basically, ARL + simplfied network (3x FCs) for rating prediction
		# The only purpose of this is to obtain the pretrained weights for ARL
		elif(self.args.model == "ANRS"):

			# Rating Prediction using 3x FCs
			rating_pred = self.ANRS_RatingPred(userAspDoc, itemAspDoc, verbose = verbose)


		if(verbose > 0):
			tqdm.write("\n[Final Output of {}] rating_pred: {}\n".format( self.args.model, rating_pred.size() ))

		return rating_pred


