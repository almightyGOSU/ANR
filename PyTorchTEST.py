import torch
import torch.nn as nn

from model.utilities import *
from model.ModelZoo import ModelZoo
from model.Logger import Logger
from model.Timer import Timer

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import argparse
from datetime import datetime



parser = argparse.ArgumentParser()

# Dataset & Model
parser.add_argument("-d", 		dest = "dataset", 	type = str, default = "musical_instruments", 	help = "Dataset for Running Experiments (Default: musical_instruments)")
parser.add_argument("-m", 		dest = "model", 	type = str, default = "ANR", 					help = "Model Name, e.g. DeepCoNN|DAttn|ANR|ANRS (Default: ANR)")

# General Hyperparameters
parser.add_argument("-bs", 			dest = "batch_size", 	type = int, 	default = 128,	 		help = "Batch Size (Default: 128)")
parser.add_argument("-e", 			dest = "epochs", 		type = int, 	default = 25, 			help = "Number of Training Epochs (Default: 25)")
parser.add_argument("-lr", 			dest = "learning_rate", type = float, 	default = 2E-3, 		help = "Learning Rate (Default: 0.002, i.e 2E-3)")
parser.add_argument("-opt", 		dest = "optimizer", 	type = str, 	default = "Adam", 		help = "Optimizer, e.g. Adam|RMSProp|SGD (Default: Adam)")
parser.add_argument("-loss_func", 	dest = "loss_function", type = str, 	default = "MSELoss", 	help = "Loss Function, e.g. MSELoss|L1Loss (Default: MSELoss)")
parser.add_argument("-dr",			dest = "dropout_rate", 	type = float, 	default = 0.5, 			help = "Dropout rate (Default: 0.5)")

# Dataset-Specific Settings (Document Length, Vocabulary Size, Dimensionality of the Embedding Layer, Source of Pretrained Word Embeddings)
parser.add_argument("-MDL", 		dest = "max_doc_len", 		type = int, 	default = 500, 		help = "Maximum User/Item Document Length (Default: 500)")
parser.add_argument("-v", 			dest = "vocab_size", 		type = int, 	default = 50000, 	help = "Vocabulary Size (Default: 50000)")
parser.add_argument("-WED", 		dest = "word_embed_dim", 	type = int, 	default = 300, 		help = "Number of Dimensions for the Word Embeddings (Default: 300)")
parser.add_argument("-p", 			dest = "pretrained_src", 	type = int, 	default = 1,		help = "Source of Pretrained Word Embeddings? \
	0: Randomly Initialized (Random Uniform Dist. from [-0.01, 0.01]), 1: w2v (Google News, 300d), 2: GloVe (6B, 400K, 100d) (Default: 1)")


# ANR Hyperparameters
parser.add_argument("-K", 		dest = "num_aspects", 	type = int, 	default = 5, 	help = "Number of Aspects (Default: 5)")
parser.add_argument("-h1", 		dest = "h1", 			type = int, 	default = 10, 	help = "Dimensionality of the Aspect-level Representations (Default: 10)")
parser.add_argument("-c", 		dest = "ctx_win_size", 	type = int, 	default = 3, 	help = "Window Size (i.e. Number of Words) for Calculating Attention (Default: 3)")
parser.add_argument("-h2", 		dest = "h2", 			type = int, 	default = 50, 	help = "Dimensionality of the Hidden Layers used for Aspect Importance Estimation (Default: 50)")
parser.add_argument("-L2_reg", 	dest = "L2_reg", 		type = float, 	default = 1E-6, help = "L2 Regularization for User & Item Bias (Default: 1E-6)")

# ANR Pretraining
parser.add_argument("-ARL_path", 	dest = "ARL_path", 	type = str, 	default = "", 	help = "Specify the file name for loading pretrained ARL weights! (Default: "", i.e. Disabled)")
parser.add_argument("-ARL_lr", 		dest = "ARL_lr", 	type = float, 	default = 0.01,	help = "RATIO of LR for fine-tuning the pretrained ARL weights (Default: 0.01)")


# Miscellaneous
parser.add_argument("-rs", 	dest = "random_seed", 			type = int, default = 1337, help = "Random Seed (Default: 1337)")
parser.add_argument("-dc", 	dest = "disable_cuda", 			type = int, default = 0, 	help = "Disable CUDA? (Default: 0, i.e. run using GPU (if available))")
parser.add_argument("-gpu", dest = "gpu", 					type = int, default = 0, 	help = "Which GPU to use? (Default: 0)")
parser.add_argument("-vb", 	dest = "verbose", 				type = int, default = 0, 	help = "Show debugging/miscellaneous information? (Default: 0, i.e. Disabled)")
parser.add_argument("-die", dest = "disable_initial_eval", 	type = int, default = 0, 	help = "Disable initial Dev/Test evaluation? (Default: 0, i.e. Disabled)")
parser.add_argument("-sm", 	dest = "save_model", 			type = str, default = "", 	help = "Specify the file name for saving model! (Default: "", i.e. Disabled)")

args = parser.parse_args()


# Check for availability of CUDA and execute on GPU if possible
args.use_cuda = not args.disable_cuda and torch.cuda.is_available()
del args.disable_cuda


# Initial Setup
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

if(args.use_cuda):
	select_gpu(args.gpu)
	torch.cuda.set_device(args.gpu)
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.deterministic = True
	torch.cuda.manual_seed(args.random_seed)
else:
	print("\n[args.use_cuda: {}] The program will be executed on the CPU!!".format( args.use_cuda ))


# Timer & Logging
timer = Timer()
timer.startTimer()

uuid = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
args.input_dir = "./datasets/{}/".format( args.dataset )
args.out_dir = "./experimental_results/{} - {}/".format( args.dataset, args.model )
log_path = "{}{}-{}".format(args.out_dir, uuid, 'logs.txt')
logger = Logger(args.out_dir, log_path, args)


# Optional: Saving Model
if(args.save_model != ""):
	saved_models_dir = "./__saved_models__/{} - {}/".format( args.dataset, args.model )
	mkdir_p(saved_models_dir)
	model_path = "{}{}_{}.pth".format( saved_models_dir, args.save_model.strip(), args.random_seed )


# Create model
mdlZoo = ModelZoo(logger, args, timer)
mdl = mdlZoo.createAndInitModel()

# Load training/validation/testing sets
train_set, train_loader, dev_set, dev_loader, test_set, test_loader = loadTrainDevTest(logger, args)
logger.log("Train/Dev/Test splits loaded! {}".format( timer.getElapsedTimeStr("init", conv2Mins = True) ))


# For evaluation
def evaluate(mdl, set_loader, epoch_num = -1, use_cuda = True, phase = "Dev", print_txt = True):

	all_rating_true = []
	all_rating_pred = []

	for batch_num, (batch_uid, batch_iid, batch_rating) in enumerate(set_loader):

		# Set to evaluation mode, important for dropout & batch normalization!
		mdl.eval()

		batch_uid = to_var(batch_uid, use_cuda = use_cuda, phase = phase)
		batch_iid = to_var(batch_iid, use_cuda = use_cuda, phase = phase)

		rating_pred = torch.squeeze(mdl(batch_uid, batch_iid))

		all_rating_true.extend(batch_rating)
		all_rating_pred.extend(rating_pred.data)

	MSE = mean_squared_error(all_rating_true, all_rating_pred)
	MAE = mean_absolute_error(all_rating_true, all_rating_pred)
	logger.log("[{}] {:6s} MSE: {:.5f}, MAE: {:.5f}".format( "Epoch {:d}".format( epoch_num + 1 ) if epoch_num >= 0 else "Initial",
		"[{}]".format( phase ), MSE, MAE), print_txt = print_txt)

	return MSE, MAE


# Initial Evaluation - Validation & Testing
if(not args.disable_initial_eval):

	logger.log("\nPerforming initial evaluation for VALIDATION set..")
	evaluate(mdl, dev_loader, use_cuda = args.use_cuda, phase = "Dev")
	logger.log("\nPerforming initial evaluation for TESTING set..")
	evaluate(mdl, test_loader, use_cuda = args.use_cuda, phase = "Test")
	logger.log("\nInitial Evaluation Complete.. {}".format( timer.getElapsedTimeStr("init", conv2Mins = True) ))


# Loss Function, Custom Regularizers, Optimizer
criterion = mdlZoo.selectLossFunction(loss_function = args.loss_function)
opt = mdlZoo.selectOptimizer(optimizer = args.optimizer, learning_rate = args.learning_rate, L2_reg = args.L2_reg)
logger.log("\nOptimizer: {}, Loss Function: {}".format( mdlZoo.optimizer, mdlZoo.loss_function ))


# Model Information
generate_mdl_summary(mdl, logger)


lstTrainingLoss = []
lstDevMSE = []
lstTestMSE = []
lstTestMAE = []

timer.startTimer("training")
for epoch_num in range(args.epochs):

	# Training loop, using mini-batches
	print("\n")
	losses = []
	for batch_num, (batch_uid, batch_iid, batch_rating) in enumerate(tqdm(train_loader, "Epoch {:d}".format( epoch_num + 1 ))):

		# Set to training mode, zero out the gradients
		mdl.train()
		opt.zero_grad()

		batch_uid = to_var(batch_uid, use_cuda = args.use_cuda)
		batch_iid = to_var(batch_iid, use_cuda = args.use_cuda)
		rating_true = to_var(batch_rating, use_cuda = args.use_cuda)

		rating_pred = torch.squeeze(mdl(batch_uid, batch_iid, verbose = (args.verbose > 0 and epoch_num == 0 and batch_num == 0)))

		loss = criterion(rating_pred.float(), rating_true.float())

		loss.backward()
		opt.step()

		losses.append(loss.data[0])


	trainingLoss = np.mean(losses)
	lstTrainingLoss.append( trainingLoss )
	logger.log("\n[Epoch {:d}/{:d}] Training Loss: {:.5f}\t{}".format(
		epoch_num + 1, args.epochs, trainingLoss, timer.getElapsedTimeStr("training", conv2HrsMins = True) ))

	# Evaluation - Validation & Testing
	devMSE, _ = evaluate(mdl, dev_loader, epoch_num = epoch_num, use_cuda = args.use_cuda, phase = "Dev")
	testMSE, testMAE = evaluate(mdl, test_loader, epoch_num = epoch_num, use_cuda = args.use_cuda, phase = "Test")

	lstDevMSE.append( devMSE )
	lstTestMSE.append( testMSE )
	lstTestMAE.append( testMAE )


	# Optional: Saving Model
	if(args.save_model != ""):

		# First, determine if this model is the best we've seen so far (i.e. having obtained the best generalization performance, based on validation MSE)
		modelIsBest = isclose(devMSE, np.min(lstDevMSE))

		if(modelIsBest):

			logger.log("\n*** MODEL has obtained the best DEV MSE of {:.5f} so far!".format( devMSE ))

			# Filter away uid_userDoc, iid_itemDoc, wid_wEmbed (These are always provided as the input, i.e. no point saving them)
			filter_list = ["uid_userDoc.weight", "iid_itemDoc.weight", "wid_wEmbed.weight"]
			filtered_mdl_state_dict = mdl.state_dict()
			filtered_mdl_state_dict = {k: v for k, v in filtered_mdl_state_dict.items() if k not in filter_list}

			model_states = {
				"best_epoch": epoch_num + 1,
				"mdl": filtered_mdl_state_dict,
				"opt": opt.state_dict()
			}
			torch.save(model_states, model_path)
			logger.log("*** MODEL saved to \"{}\"".format( model_path ))


logger.log("\n[Training Loss]\n{}".format( [float("{:.5f}".format(i)) for i in lstTrainingLoss] ))
logger.log("\n[Dev MSE]\n{}".format( [float("{:.5f}".format(i)) for i in lstDevMSE] ))
logger.log("[Test MSE]\n{}".format( [float("{:.5f}".format(i)) for i in lstTestMSE] ))
logger.log("[Test MAE]\n{}\n".format( [float("{:.5f}".format(i)) for i in lstTestMAE] ))

epoch_num_forBestDevMSE, bestDevMSE, testMSE_forBestDevMSE, testMAE_forBestDevMSE = getBestPerf(lstDevMSE, lstTestMSE, lstTestMAE)
logger.log("\nBest Dev MSE: {:.5f} (Obtained during Evaluation #{:d})".format( bestDevMSE, epoch_num_forBestDevMSE ))
logger.log("Test MSE: {:.5f}, Test MAE: {:.5f}".format( testMSE_forBestDevMSE, testMAE_forBestDevMSE ))

logger.log("\nEnd of Program! {}".format( timer.getElapsedTimeStr(conv2HrsMins = True) ))
print("\n\n\n")


