import torch
import torch.nn as nn
import torch.nn.functional as F

from .utilities import to_var

from tqdm import tqdm



'''
Aspect-based Representation Learning (ARL)
'''
class ANR_ARL(nn.Module):

	def __init__(self, logger, args):

		super(ANR_ARL, self).__init__()

		self.logger = logger
		self.args = args

		# Aspect Embeddings
		self.aspEmbed = nn.Embedding(self.args.num_aspects, self.args.ctx_win_size * self.args.h1)
		self.aspEmbed.weight.requires_grad = True

		# Aspect-Specific Projection Matrices
		self.aspProj = nn.Parameter(torch.Tensor(self.args.num_aspects, self.args.word_embed_dim, self.args.h1), requires_grad = True)

		# Initialize all weights using random uniform distribution from [-0.01, 0.01]
		self.aspEmbed.weight.data.uniform_(-0.01, 0.01)
		self.aspProj.data.uniform_(-0.01, 0.01)


	'''
	[Input]		batch_docIn:	bsz x max_doc_len x word_embed_dim
	[Output]	batch_aspRep:	bsz x num_aspects x h1
	'''
	def forward(self, batch_docIn, verbose = 0):

		if(verbose > 0):
			tqdm.write("\n============================== Aspect Representation Learning (ARL) ==============================")
			tqdm.write("[Input] batch_docIn: {}".format( batch_docIn.size() ))

		# Loop over all aspects
		lst_batch_aspAttn = []
		lst_batch_aspRep = []
		for a in range(self.args.num_aspects):

			if(verbose > 0 and a == 0):
				tqdm.write("\nAs an example, for <Aspect {}>:\n".format( a ))

			# Aspect-Specific Projection of Input Word Embeddings: (bsz x max_doc_len x h1)
			batch_aspProjDoc = torch.matmul(batch_docIn, self.aspProj[a])

			if(verbose > 0 and a == 0):
				tqdm.write("\tbatch_docIn: {}".format( batch_docIn.size() ))
				tqdm.write("\tself.aspProj[{}]: {}".format( a, self.aspProj[a].size() ))
				tqdm.write("\tbatch_aspProjDoc: {}".format( batch_aspProjDoc.size() ))


			# Aspect Embedding: (bsz x h1 x 1) after tranposing!
			bsz = batch_docIn.size()[0]
			batch_aspEmbed = self.aspEmbed( to_var(torch.LongTensor(bsz, 1).fill_(a), use_cuda = self.args.use_cuda) )
			batch_aspEmbed = torch.transpose(batch_aspEmbed, 1, 2)
			if(verbose > 0 and a == 0):
				tqdm.write("\n\tbatch_aspEmbed: {}".format( batch_aspEmbed.size() ))


			# Window Size (self.args.ctx_win_size) of 1: Calculate Attention based on the word itself!
			if(self.args.ctx_win_size == 1):

				# Calculate Attention: Inner Product & Softmax
				# (bsz x max_doc_len x h1) x (bsz x h1 x 1) -> (bsz x max_doc_len x 1)
				batch_aspAttn = torch.matmul(batch_aspProjDoc, batch_aspEmbed)
				batch_aspAttn = F.softmax(batch_aspAttn, dim = 1)
				if(verbose > 0 and a == 0):
					tqdm.write("\n\tbatch_aspAttn: {}".format( batch_aspAttn.size() ))

			# Context-based Word Importance
			# Calculate Attention based on the word itself, and the (self.args.ctx_win_size - 1) / 2 word(s) before & after it
			else:

				# Pad the document
				pad_size = int( (self.args.ctx_win_size - 1) / 2 )
				batch_aspProjDoc_padded = F.pad(batch_aspProjDoc, (0, 0, pad_size, pad_size), "constant", 0)
				if(verbose > 0 and a == 0):
					tqdm.write("\n\tbatch_aspProjDoc_padded [PADDED; Pad Size: {}]: {}".format( pad_size, batch_aspProjDoc_padded.size() ))

				# Use "sliding window" using stride of 1 (word at a time) to generate word chunks of ctx_win_size
				# (bsz x max_doc_len x h1) -> (bsz x max_doc_len x (ctx_win_size x h1))
				batch_aspProjDoc_padded = batch_aspProjDoc_padded.unfold(1, self.args.ctx_win_size, 1)
				if(verbose > 0 and a == 0):
					tqdm.write("\tbatch_aspProjDoc_padded: {}".format( batch_aspProjDoc_padded.size() ))
				batch_aspProjDoc_padded = torch.transpose(batch_aspProjDoc_padded, 2, 3)
				if(verbose > 0 and a == 0):
					tqdm.write("\tbatch_aspProjDoc_padded: {}".format( batch_aspProjDoc_padded.size() ))
				batch_aspProjDoc_padded = batch_aspProjDoc_padded.contiguous().view(-1, self.args.max_doc_len, self.args.ctx_win_size * self.args.h1)
				if(verbose > 0 and a == 0):
					tqdm.write("\tbatch_aspProjDoc_padded: {}".format( batch_aspProjDoc_padded.size() ))

				# Calculate Attention: Inner Product & Softmax
				# (bsz x max_doc_len x (ctx_win_size x h1)) x (bsz x (ctx_win_size x h1) x 1) -> (bsz x max_doc_len x 1)
				batch_aspAttn = torch.matmul(batch_aspProjDoc_padded, batch_aspEmbed)

				batch_aspAttn = F.softmax(batch_aspAttn, dim = 1)
				if(verbose > 0 and a == 0):
					tqdm.write("\n\tbatch_aspAttn [Window Size: {}]: {}".format( self.args.ctx_win_size, batch_aspAttn.size() ))


			# Weighted Sum: Broadcasted Element-wise Multiplication & Sum over Words
			# (bsz x max_doc_len x 1) and (bsz x max_doc_len x h1) -> (bsz x h1)
			batch_aspRep = batch_aspProjDoc * batch_aspAttn.expand_as(batch_aspProjDoc)
			if(verbose > 0 and a == 0):
				tqdm.write("\n\tbatch_aspRep: {}".format( batch_aspRep.size() ))
			batch_aspRep = torch.sum(batch_aspRep, dim = 1)
			if(verbose > 0 and a == 0):
				tqdm.write("\tbatch_aspRep: {}".format( batch_aspRep.size() ))


			# Store the results (Attention & Representation) for this aspect
			lst_batch_aspAttn.append(torch.transpose(batch_aspAttn, 1, 2))
			lst_batch_aspRep.append(torch.unsqueeze(batch_aspRep, 1))


		# Reshape the Attentions & Representations
		# batch_aspAttn:	(bsz x num_aspects x max_doc_len)
		# batch_aspRep:		(bsz x num_aspects x h1)
		batch_aspAttn = torch.cat(lst_batch_aspAttn, dim = 1)
		batch_aspRep = torch.cat(lst_batch_aspRep, dim = 1)

		if(verbose > 0):
			tqdm.write("\n[Output] <All {} Aspects>".format( self.args.num_aspects ))
			tqdm.write("[Output] batch_aspAttn: {}".format( batch_aspAttn.size() ))
			tqdm.write("[Output] batch_aspRep: {}".format( batch_aspRep.size() ))
			tqdm.write("============================== ==================================== ==============================\n")


		# Returns the aspect-level attention over document words, and the aspect-based representations
		return batch_aspAttn, batch_aspRep

