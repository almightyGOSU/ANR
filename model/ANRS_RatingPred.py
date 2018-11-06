import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm



'''
Rating Prediction using the 'Simplified Model', i.e. 3x FCs
The only purpose of this is to obtain the pretrained weights for ARL
'''
class ANRS_RatingPred(nn.Module):

	def __init__(self, logger, args):

		super(ANRS_RatingPred, self).__init__()

		self.logger = logger
		self.args = args


		# User/Item FC to learn the abstract user & item representations, respectively
		self.userFC = nn.Linear(self.args.num_aspects * self.args.h1, self.args.h1)
		self.itemFC = nn.Linear(self.args.num_aspects * self.args.h1, self.args.h1)

		# Dropout, using the specified dropout probability
		self.userFC_Dropout = nn.Dropout(p = self.args.dropout_rate)
		self.itemFC_Dropout = nn.Dropout(p = self.args.dropout_rate)

		# Dimensionality of the abstract user & item representations
		# Well, this can also be a hyperparameter, but we simply set it to h1
		self.user_item_rep_dim = self.args.h1

		# Prediction Layer
		self.prediction = nn.Linear(2 * self.user_item_rep_dim, 1)

		# Initialize all weights using random uniform distribution from [-0.01, 0.01]
		self.userFC.weight.data.uniform_(-0.01, 0.01)
		self.itemFC.weight.data.uniform_(-0.01, 0.01)
		self.prediction.weight.data.uniform_(-0.01, 0.01)


	'''
	[Input]	userAspRep:		bsz x num_aspects x h1
	[Input]	itemAspRep:		bsz x num_aspects x h1
	'''
	def forward(self, userAspRep, itemAspRep, verbose = 0):

		if(verbose > 0):
			tqdm.write("\n\n============================== Aspect-Based BASIC Rating Predictor ==============================")
			tqdm.write("[Input] userAspRep: {}".format( userAspRep.size() ))
			tqdm.write("[Input] itemAspRep: {}".format( itemAspRep.size() ))


		# Concatenate all aspect-level representations into a single vector
		concatUserRep = userAspRep.view(-1, self.args.num_aspects * self.args.h1)
		concatItemRep = itemAspRep.view(-1, self.args.num_aspects * self.args.h1)

		if(verbose > 0):
			tqdm.write("\n[Concatenated] concatUserRep: {}".format( concatUserRep.size() ))
			tqdm.write("[Concatenated] concatItemRep: {}".format( concatItemRep.size() ))


		# Fully-Connected (To get the abstract user & item representations)
		abstractUserRep = self.userFC(concatUserRep)
		abstractItemRep = self.itemFC(concatItemRep)

		if(verbose > 0):
			tqdm.write("\n[After FC, i.e. torch.nn.Linear] abstractUserRep: {}".format( abstractUserRep.size() ))
			tqdm.write("[After FC, i.e. torch.nn.Linear] abstractItemRep: {}".format( abstractItemRep.size() ))

		# Non-Linearity: ReLU
		abstractUserRep = F.relu(abstractUserRep)
		abstractItemRep = F.relu(abstractItemRep)

		if(verbose > 0):
			tqdm.write("[After ReLU] abstractUserRep: {}".format( abstractUserRep.size() ))
			tqdm.write("[After ReLU] abstractItemRep: {}".format( abstractItemRep.size() ))

		# Dropout
		abstractUserRep = self.userFC_Dropout(abstractUserRep)
		abstractItemRep = self.itemFC_Dropout(abstractItemRep)

		if(verbose > 0):
			tqdm.write("[After Dropout (Dropout Rate of {:.1f})] abstractUserRep: {}".format( self.args.dropout_rate, abstractUserRep.size() ))
			tqdm.write("[After Dropout (Dropout Rate of {:.1f})] abstractItemRep: {}".format( self.args.dropout_rate, abstractItemRep.size() ))


		# Concatenate the user & item representations for prediction
		userItemRep = torch.cat((abstractUserRep, abstractItemRep), 1)
		if(verbose > 0):
			tqdm.write("\n[Input to Final Prediction Layer] userItemRep: {}".format( userItemRep.size() ))

		# Actual Rating Prediction
		# FC: Fully Connected, i.e. torch.nn.Linear
		rating_pred = self.prediction(userItemRep)


		if(verbose > 0):
			tqdm.write("\n[ANRS_RatingPred Output] rating_pred: {}".format( rating_pred.size() ))
			tqdm.write("============================== =================================== ==============================\n")

		return rating_pred


