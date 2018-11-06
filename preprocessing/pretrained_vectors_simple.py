from preprocessing_simple_utilities import *
import gensim



parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type = str, default = "amazon_instant_video", help = "Dataset (Default: amazon_instant_video)")

parser.add_argument("-emb_dim", "--emb_dim", type = int, default = 300, help = "Embeddings Dimension (Default: 300)")
parser.add_argument("-emb_rand_init", "--emb_rand_init", type = float, metavar = "<float>", default = 0.01,
	help = "Random Initialization of Embeddings for Words without Pretrained Embeddings (Default: 0.01)")

parser.add_argument("-rs", '--random_seed', dest = "random_seed", type = int, metavar = "<int>", default = 1337, help = 'Random seed (Default: 1337)')

args = parser.parse_args()
args.dataset = args.dataset.strip()



# Dataset, e.g. amazon_instant_video
CATEGORY = args.dataset.strip().lower()
print("\nDataset: {}".format( CATEGORY ))



# Random seed
np.random.seed(args.random_seed)
random.seed(args.random_seed)



startTime = time.time()

# ============================================================================= INPUT =============================================================================
# Source Folder
SOURCE_FOLDER		= "../datasets/"

# Category Folder
CATEGORY_FOLDER		= "{}{}/".format( SOURCE_FOLDER, CATEGORY )

# Environment (Contains word_wid, i.e. the mapping from words in the vocabulary to their wid)
input_env			= "{}{}{}".format( CATEGORY_FOLDER, CATEGORY, fp_env )

# NOTE: This is the input file for the pretrained word embeddings
# NOTE: Specify the correct file!!
input_embeddings	= "../../GoogleNews-vectors-negative300.bin"
# ============================================================================= INPUT =============================================================================



# ============================================================================= OUTPUT =============================================================================
# This contains the embeddings of words (within vocabulary) that can be found in the pretrained embeddings
output_wid_wordEmbed	= "{}{}{}".format( CATEGORY_FOLDER, CATEGORY, fp_wid_wordEmbed )

output_log				= "{}{}{}".format( CATEGORY_FOLDER, CATEGORY, "___pretrained_vectors_log.txt" )
# ============================================================================= OUTPUT =============================================================================



# Clear the embedding file (if present)
with open(output_wid_wordEmbed, 'w+') as f:
	f.write('')

# Clear the info (preprocessing log) file
with open(output_log, 'w+') as f:
	f.write('')


# Arguments Info
print("{}".format(""))
append_to_file(output_log, print_args(args))

# Input Info
append_to_file(output_log, "\n{:<38s} {}".format( "[INPUT] Source Folder:", SOURCE_FOLDER ))
append_to_file(output_log, "{:<38s} {}".format( "[INPUT] Category Folder:", CATEGORY_FOLDER ))
append_to_file(output_log, "{:<38s} {}".format( "[INPUT] env:", input_env ))
append_to_file(output_log, "{:<38s} {}".format( "[INPUT] Pretrained Word Embeddings:", input_embeddings ))

# Output Info
append_to_file(output_log, "\n{:<38s} {}".format( "[OUTPUT] wid_wordEmbed:", output_wid_wordEmbed ))



# Load word-to-wid mappings from the environment file
append_to_file(output_log, "\nLoading word-to-wid mappings (i.e. word_wid) from \"{}\"".format( input_env ))
env = load_pickle( input_env )
word_wid = env['word_wid']
wid_word = {wid: word for word, wid in word_wid.items()}

# Force garbage collection
del env
gc.collect()


# Vocab
vocab = word_wid.keys()
append_to_file(output_log, "\n|V|: {}".format( len(vocab) ))



embeddings = {}

# Load word embeddings
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(input_embeddings, binary = True)
w2v_vocab = w2v_model.vocab.keys()
w2v_vocab_dict = {word: "" for word in w2v_vocab}
for v in tqdm(vocab, "Processing pretrained vectors"):
	try:
		_ = w2v_vocab_dict[v]
		embeddings[v] = w2v_model[v]
	except:
		pass

append_to_file(output_log, "\nFinished processing pretrained embeddings..")
append_to_file(output_log, "# of words with pretrained embeddings: {}".format( len(embeddings) ))


wordEmbedMatrix = []
noPretrainedEmb_words = []

# For '<pad>' and '<unk>'
wordEmbedMatrix.append(np.zeros((args.emb_dim)).tolist())
wordEmbedMatrix.append(np.zeros((args.emb_dim)).tolist())
noPretrainedEmb_words.append(PAD)
noPretrainedEmb_words.append(UNK)

for wid in tqdm(range(2, len(word_wid)), "Forming Matrix"):

	word = wid_word[wid]

	try:
		vec = embeddings[word]
		vec = vec.tolist()
		vec = [float(x) for x in vec]

		wordEmbedMatrix.append(vec)

	except:
		# Words that do not have a pretrained embedding are initialized randomly using a uniform distribution U(−0.01, 0.01)
		vec = np.random.uniform(low = -args.emb_rand_init, high = args.emb_rand_init, size = (args.emb_dim)).tolist()

		wordEmbedMatrix.append(vec)
		noPretrainedEmb_words.append(word)

wordEmbedMatrix = np.stack(wordEmbedMatrix)
wordEmbedMatrix = np.reshape(wordEmbedMatrix, (len(word_wid), args.emb_dim))


noPretrainedEmb_words.sort()
append_to_file(output_log, "\nEmbedding Matrix = {}, |V| = {}".format( wordEmbedMatrix.shape, len(word_wid) ))
append_to_file(output_log, "# of Words w/ No Pretrained Embeddings = {} ({:.3f}%)\n".format(
	len(noPretrainedEmb_words), (len(noPretrainedEmb_words) / len(word_wid) * 100) ))


np.save(output_wid_wordEmbed, wordEmbedMatrix)
append_to_file(output_log, "Embeddings successfully saved to '{}'\n".format( output_wid_wordEmbed ))


# Force garbage collection
gc.collect()

endTime = time.time()
durationInSecs = endTime - startTime
durationInMins = durationInSecs / 60
append_to_file(output_log, "\nPretrained word embeddings for \"{}\" obtained after {:.2f} seconds ({:.2f} minutes)\n".format(
	CATEGORY, durationInSecs, durationInMins))


