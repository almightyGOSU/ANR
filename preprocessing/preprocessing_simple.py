from preprocessing_simple_utilities import *
from itertools import groupby



parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type = str, default = "amazon_instant_video", help = "Dataset (Default: amazon_instant_video)")

parser.add_argument("-dmax", "--dataset_maximum_size", type = int, default = 5000000, help = "Maximum Size of Dataset, Randomly subsample if larger than this. (Default: 5,000,000)")

parser.add_argument("-minRL", "--minRL", type = int, default = 10, help = "Minimum Review Length (Default: 10)")
parser.add_argument("-minReviews", "--minReviews", type = int, default = 1, help = "Minimum Reviews Per User/Item (Default: 1)")
parser.add_argument("-v", "--vocab", type = int, default = 50000, help = "Vocabulary Size (Default: 50000)")
parser.add_argument("-maxDL", "--maxDL", type = int, default = 500, help = "Maximum Document Length (Default: 500)")

parser.add_argument("-tr", "--train_ratio", dest = "train_ratio", type = float, metavar = "<float>", default = 0.8, help = "Train Ratio (Default: 0.8)")
parser.add_argument("-rs", '--random_seed', dest = "random_seed", type = int, metavar = "<int>", default = 1337, help = 'Random seed (Default: 1337)')

parser.add_argument("-dev_test_in_train", dest = "dev_test_in_train", type = int, metavar = "<int>", default = 0,
	help = "To remove users/items that do not appear during training from the DEV and TEST sets (Default: 0, i.e. False)")

args = parser.parse_args()



# "Convert" the "0/1" to "False/True"
args.dev_test_in_train = True if (args.dev_test_in_train == 1) else False



# Dataset, e.g. amazon_instant_video
CATEGORY = args.dataset.strip().lower()
print("\nDataset: {}".format( CATEGORY ))


MIN_REVIEW_LEN = args.minRL 		# Minimum review length based on number of tokens
MIN_REVIEWS = args.minReviews		# Minimum number of reviews per user/item
VOCAB_SIZE = args.vocab				# Vocabulary size
MAX_DOC_LEN = args.maxDL			# Maximum length of user/item document



# Random seed
np.random.seed(args.random_seed)
random.seed(args.random_seed)



startTime = time.time()

# ============================================================================= INPUT =============================================================================
# Source Folder
SOURCE_FOLDER = "../datasets/"

# Input Reviews (Each review has a corresponding rating, user, and item)
REVIEW_JSON = "{}{}.json".format( SOURCE_FOLDER, CATEGORY )
# ============================================================================= INPUT =============================================================================



# ============================================================================= OUTPUT =============================================================================
# Category Folder
CATEGORY_FOLDER = "{}{}/".format( SOURCE_FOLDER, CATEGORY )
try:
	os.makedirs(CATEGORY_FOLDER)
except:
	pass

output_log			= "{}{}{}".format( CATEGORY_FOLDER, CATEGORY, "___preprocessing_log.txt" )

output_env			= "{}{}{}".format( CATEGORY_FOLDER, CATEGORY, fp_env )
output_info			= "{}{}{}".format( CATEGORY_FOLDER, CATEGORY, fp_info )
output_split_train	= "{}{}{}".format( CATEGORY_FOLDER, CATEGORY, fp_split_train )
output_split_dev	= "{}{}{}".format( CATEGORY_FOLDER, CATEGORY, fp_split_dev )
output_split_test	= "{}{}{}".format( CATEGORY_FOLDER, CATEGORY, fp_split_test )
output_uid_userDoc	= "{}{}{}".format( CATEGORY_FOLDER, CATEGORY, fp_uid_userDoc )
output_iid_itemDoc	= "{}{}{}".format( CATEGORY_FOLDER, CATEGORY, fp_iid_itemDoc )
# ============================================================================= OUTPUT =============================================================================



# Clear the info (preprocessing log) file
with open(output_log, 'w+') as f:
	f.write('')

# Arguments Info
append_to_file(output_log, print_args(args))

# Input Info
append_to_file(output_log, "\n{:<28s} {}".format( "[INPUT] Source Folder:", SOURCE_FOLDER ))
append_to_file(output_log, "{:<28s} {}".format( "[INPUT] Reviews/Ratings:", REVIEW_JSON ))

# Output Info
append_to_file(output_log, "\n{:<28s} {}".format( "[OUTPUT] Category Folder:", CATEGORY_FOLDER ))
append_to_file(output_log, "{:<28s} {}".format( "[OUTPUT] env:", output_env ))
append_to_file(output_log, "{:<28s} {}".format( "[OUTPUT] info:", output_info ))
append_to_file(output_log, "{:<28s} {}".format( "[OUTPUT] split_train:", output_split_train ))
append_to_file(output_log, "{:<28s} {}".format( "[OUTPUT] split_dev:", output_split_dev ))
append_to_file(output_log, "{:<28s} {}".format( "[OUTPUT] split_test:", output_split_test ))
append_to_file(output_log, "{:<28s} {}".format( "[OUTPUT] uid_userDoc:", output_uid_userDoc ))
append_to_file(output_log, "{:<28s} {}".format( "[OUTPUT] iid_itemDoc:", output_iid_itemDoc ))


print("\nPreprocessing data for \"{}\"".format( CATEGORY ))

append_to_file(output_log, "\n[Settings]\nMin reviews for user/item: {}".format( MIN_REVIEWS ))
append_to_file(output_log, "Min review length to qualify as an user-item interaction: {}".format( MIN_REVIEW_LEN ))
append_to_file(output_log, "Max words for user/item document: {} (For truncating/padding to get a fixed-size representation)".format( MAX_DOC_LEN ))
append_to_file(output_log, "Top-{} words in vocabulary being utilized!\n".format( VOCAB_SIZE ))



interactions = []

# Initial pass of reviews to get the user-item interactions
append_to_file(output_log, "\nInitial pass of reviews to get the user-item interactions!")
with codecs.open(REVIEW_JSON, 'r', encoding = 'utf-8', errors = 'ignore') as inFile:

	lines = inFile.readlines()

	for line in tqdm(lines, "Initial pass of reviews for \"{}\"".format( CATEGORY )):

		d = json.loads(line)

		if("Yelp" in CATEGORY):
			user = d['user_id']
			item = d['business_id']
		else:
			user = d['reviewerID']
			item = d['asin']

		interactions.append([user, item])

user_count, item_count = count(interactions)
num_reviews = len(interactions)

# Force garbage collection
lines.clear()
gc.collect()

append_to_file(output_log, "[Initial stats] Users: {:,}, Items: {:,}, Ratings: {:,}, Density: {:.7f}\n".format(
	len(user_count), len(item_count), num_reviews, float(num_reviews) / (len(user_count) * len(item_count)) ))


print("\nStarting to filter away users & items based on thresold of {} reviews!".format( MIN_REVIEWS ))
while True:

	user_count, item_count = count(interactions)
	oldUsers = len(list(user_count))
	oldItems = len(list(item_count))

	# Drop users with less than "MIN_REVIEWS" reviews
	drop_if_lt(user_count, MIN_REVIEWS)

	# Drop items with less than "MIN_REVIEWS" reviews
	drop_if_lt(item_count, MIN_REVIEWS)

	currUsers = len(list(user_count))
	currItems = len(list(item_count))
	print("\nFiltered users & items based on thresold of {} reviews!".format(MIN_REVIEWS))
	print("Users: {} -> {}, Items: {} -> {}".format(oldUsers, currUsers, oldItems, currItems))

	# Check if no users and items were filtered
	if(oldUsers == currUsers and oldItems == currItems):
		append_to_file(output_log, "\nNo change in # of users or # of items!\n")
		append_to_file(output_log, "[Final stats] Users: {:,}, Items: {:,}, Ratings: {:,}, Density: {:.7f}".format(
			len(user_count), len(item_count), num_reviews, float(num_reviews) / (len(user_count) * len(item_count)) ))
		break

	# Update interactions based on user & item Counters
	print("Updating interactions based on remaining users & items..")
	users = sorted(list(user_count))
	items = sorted(list(item_count))
	users_dict = {user: "" for user in users}
	items_dict = {item: "" for item in items}
	interactions[:] = [interaction for interaction in tqdm(interactions, "Updating interactions") 
		if (hit(interaction, users_dict, items_dict) == True)]

	user_count, item_count = count(interactions)
	num_reviews = len(interactions)

	append_to_file(output_log, "[Current stats] Users: {}, Items: {}, Ratings: {}, Density: {:.7f}".format(
		len(user_count), len(item_count), num_reviews, float(num_reviews) / (len(user_count) * len(item_count)) ))

	# Progress update
	currTime = time.time()
	elapsedTimeInSecs = currTime - startTime
	elapsedTimeInMins = elapsedTimeInSecs / 60
	print("\nElapsed time for \"{}\": {:.2f} seconds ({:.2f} minutes)".format( CATEGORY, elapsedTimeInSecs, elapsedTimeInMins) )



# Users & items satisfying the minimum number of reviews (MIN_REVIEWS)
users = sorted(list(user_count))
items = sorted(list(item_count))
users_dict = {user: "" for user in users}
items_dict = {item: "" for item in items}

# Force garbage collection
interactions.clear()
gc.collect()



interactions = []

# Second pass of reviews to get the rating, date, and tokenized review
append_to_file(output_log, "\n\nSecond pass of reviews to get the rating, date, and tokenized review!")
with codecs.open(REVIEW_JSON, 'r', encoding = 'utf-8', errors = 'ignore') as inFile:

	lines = inFile.readlines()

	for line in tqdm(lines, "Second pass of reviews for \"{}\"".format( CATEGORY )):

		d = json.loads(line)

		if("Yelp" in CATEGORY):
			user = d['user_id']
			item = d['business_id']
		else:
			user = d['reviewerID']
			item = d['asin']

		if (hit( (user, item), users_dict, items_dict )):

			if("Yelp" in CATEGORY):
				rating = d['stars']
				date = d['date']
				date = int(time.mktime(time.strptime(date, "%Y-%m-%d")))
				text = simple_tokenizer(d['text'])
			else:
				rating = d['overall']
				date = d['unixReviewTime']
				text = simple_tokenizer(d['reviewText'])

			interactions.append([user, item, rating, date, text])

user_count, item_count = count(interactions)
num_reviews = len(interactions)

# Force garbage collection
users.clear()
items.clear()
lines.clear()
gc.collect()

append_to_file(output_log, "[Current stats] Users: {:,}, Items: {:,}, Ratings: {:,}, Density: {:.7f}".format(
		len(user_count), len(item_count), num_reviews, float(num_reviews) / (len(user_count) * len(item_count)) ))



# BEFORE filtering reviews with less than MIN_REVIEW_LEN tokens
oldUsers = len(list(user_count))
oldItems = len(list(item_count))

# Filter user-item interactions based on minimum review length
append_to_file(output_log, "\nFiltering user-item interactions based on minimum review length of {} tokens..".format( MIN_REVIEW_LEN ))
interactions[:] = [interaction for interaction in tqdm(interactions, "Filtering interactions") 
	if (len(interaction[4]) >= MIN_REVIEW_LEN)]

user_count, item_count = count(interactions)
num_reviews = len(interactions)

# AFTER filtering reviews with less than MIN_REVIEW_LEN tokens
currUsers = len(list(user_count))
currItems = len(list(item_count))
print("\nFiltered users & items based on minimum review length of {} tokens!".format( MIN_REVIEW_LEN ))
print("Users: {:,} -> {:,}, Items: {:,} -> {:,}".format(oldUsers, currUsers, oldItems, currItems))

append_to_file(output_log, "[Current stats] Users: {:,}, Items: {:,}, Ratings: {:,}, Density: {:.7f}\n".format(
	len(user_count), len(item_count), num_reviews, float(num_reviews) / (len(user_count) * len(item_count)) ))



print("\nStarting to filter away users & items based on thresold of {} reviews (after removing reviews with <= {} tokens)!".format( MIN_REVIEWS, MIN_REVIEW_LEN ))
while True:

	user_count, item_count = count(interactions)
	oldUsers = len(list(user_count))
	oldItems = len(list(item_count))

	# Drop users with less than "MIN_REVIEWS" reviews
	drop_if_lt(user_count, MIN_REVIEWS)

	# Drop items with less than "MIN_REVIEWS" reviews
	drop_if_lt(item_count, MIN_REVIEWS)

	currUsers = len(list(user_count))
	currItems = len(list(item_count))
	print("\nFiltered users & items based on thresold of {} reviews!".format(MIN_REVIEWS))
	print("Users: {:,} -> {:,}, Items: {:,} -> {:,}".format(oldUsers, currUsers, oldItems, currItems))

	# Check if no users and items were filtered
	if(oldUsers == currUsers and oldItems == currItems):
		append_to_file(output_log, "\nNo change in # of users or # of items!\n")
		append_to_file(output_log, "[Final stats] Users: {:,}, Items: {:,}, Ratings: {:,}, Density: {:.7f}".format(
			len(user_count), len(item_count), num_reviews, float(num_reviews) / (len(user_count) * len(item_count)) ))
		break

	# Update interactions based on user & item Counters
	print("Updating interactions based on remaining users & items..")
	users = sorted(list(user_count))
	items = sorted(list(item_count))
	users_dict = {user: "" for user in users}
	items_dict = {item: "" for item in items}
	interactions[:] = [interaction for interaction in tqdm(interactions, "Updating interactions") 
		if (hit(interaction, users_dict, items_dict) == True)]

	user_count, item_count = count(interactions)
	num_reviews = len(interactions)

	append_to_file(output_log, "[Current stats] Users: {:,}, Items: {:,}, Ratings: {:,}, Density: {:.7f}".format(
		len(user_count), len(item_count), num_reviews, float(num_reviews) / (len(user_count) * len(item_count)) ))

	# Progress update
	currTime = time.time()
	elapsedTimeInSecs = currTime - startTime
	elapsedTimeInMins = elapsedTimeInSecs / 60
	print("\nElapsed time for \"{}\": {:.2f} seconds ({:.2f} minutes)".format( CATEGORY, elapsedTimeInSecs, elapsedTimeInMins ))



# For LIMITING large datasets
if(len(interactions) > args.dataset_maximum_size):

	append_to_file(output_log, "\n{}".format( "*" * 125 ))
	append_to_file(output_log, "*** Original Dataset Size (i.e. num_ratings): {:,}!".format( len(interactions) ))
	append_to_file(output_log, "*** Selecting a random subsample of {:,} user-item interactions!".format( args.dataset_maximum_size ))
	interactions = random.sample(interactions, args.dataset_maximum_size)
	append_to_file(output_log, "*** Current Dataset Size (i.e. num_ratings):  {:,}!".format( len(interactions) ))
	append_to_file(output_log, "{}".format( "*" * 125 ))



# TRAIN/DEV/TEST ratio
dev_test_ratio = ((1.0 - args.train_ratio) / 2.0)

train_interactions = []
dev_interactions = []
test_interactions = []



# Splitting into train/dev/test
append_to_file(output_log, "{:.1f}% of ALL reviews are RANDOMLY selected for TRAIN, another {:.1f}% RANDOMLY selected for DEV, and remaining {:.1f}% used for TEST.".format(
	args.train_ratio * 100, dev_test_ratio * 100, dev_test_ratio * 100))

# Random shuffle for all user-item interactions
random.shuffle(interactions)

# Total number of reviews
num_reviews = len(interactions)

train_index = int(args.train_ratio * num_reviews)
train_interactions = interactions[:train_index]

dev_ratio = args.train_ratio + dev_test_ratio
dev_index = int(dev_ratio * num_reviews)
dev_interactions = interactions[train_index:dev_index]
test_interactions = interactions[dev_index:]


# Assertion to make sure TRAIN + DEV + TEST = ORIGINAL DATASET SIZE
num_reviews = len(interactions)
assert (len(train_interactions) + len(dev_interactions) + len(test_interactions)) == num_reviews, "Train/Dev/Test split is wrong!!"


train_pct = (len(train_interactions) * 100.00) / num_reviews
dev_pct = (len(dev_interactions) * 100.00) / num_reviews
test_pct = (len(test_interactions) * 100.00) / num_reviews
append_to_file(output_log, "\n[Initial Stats] Total Interactions: {:,}, TRAIN: {:,} ({:.2f}%), DEV: {:,} ({:.2f}%), TEST: {:,} ({:.2f}%)".format(
	num_reviews, len(train_interactions), train_pct, len(dev_interactions), dev_pct, len(test_interactions), test_pct ))




# If true (i.e. -dev_test_in_train 1), users/items in the DEV and TEST sets are supposed to be present during the training process
# NOTE: We chose to do this INSTEAD OF representing such users & items with an EMPTY document.
# NOTE: For users (or items) who do not appear in the TRAINING set, their documents will be EMPTY since we construct these documents based on reviews available from the TRAINING set.
# NOTE: Such a EMPTY document will essentially be a matrix of 0s, which is rather meaningless for any review-based model.
if(args.dev_test_in_train):

	# Remove users & items who do not appear in training set, from the dev and test sets
	append_to_file(output_log, "\n\nRemoving users & items who do not appear in the training set, from the dev and test sets..")
	train_user_count, train_item_count = count(train_interactions)

	oldDevSize = len(dev_interactions)
	oldTestSize = len(test_interactions)

	train_users = sorted(list(train_user_count))
	train_items = sorted(list(train_item_count))
	train_users_dict = {user: "" for user in train_users}
	train_items_dict = {item: "" for item in train_items}
	dev_interactions[:] = [i for i in tqdm(dev_interactions, "Updating DEV interactions") if (hit(i, train_users_dict, train_items_dict) == True)]
	test_interactions[:] = [i for i in tqdm(test_interactions, "Updating TEST interactions") if (hit(i, train_users_dict, train_items_dict) == True)]

	newDevSize = len(dev_interactions)
	newTestSize = len(test_interactions)

	append_to_file(output_log, "\nRemoved {:,} interactions from DEV and {:,} interactions from TEST! (i.e. Those belonging to Users/Items which do not appear in TRAIN)".format(
		(oldDevSize - newDevSize), (oldTestSize - newTestSize) ))

	num_reviews = len(train_interactions) + len(dev_interactions) + len(test_interactions)
	train_pct = (len(train_interactions) * 100.00) / num_reviews
	dev_pct = (len(dev_interactions) * 100.00) / num_reviews
	test_pct = (len(test_interactions) * 100.00) / num_reviews
	append_to_file(output_log, "\n[Current Stats] Total Interactions: {:,}, TRAIN: {:,} ({:.2f}%), DEV: {:,} ({:.2f}%), TEST: {:,} ({:.2f}%)".format(
		num_reviews, len(train_interactions), train_pct, len(dev_interactions), dev_pct, len(test_interactions), test_pct))



# Just for the statistics
user_count = Counter()
item_count = Counter()
stack_count(train_interactions, user_count, item_count)
stack_count(dev_interactions, user_count, item_count)
stack_count(test_interactions, user_count, item_count)
append_to_file(output_log, "\n\n\n\n[FINAL Stats] Users: {:,}, Items: {:,}, Ratings: {:,}, Density: {:.7f}\n".format(
			len(user_count), len(item_count), num_reviews, float(num_reviews) / (len(user_count) * len(item_count)) ))

num_reviews = len(train_interactions) + len(dev_interactions) + len(test_interactions)
train_pct = (len(train_interactions) * 100.00) / num_reviews
dev_pct = (len(dev_interactions) * 100.00) / num_reviews
test_pct = (len(test_interactions) * 100.00) / num_reviews
append_to_file(output_log, "[FINAL Stats] Total Interactions: {:,}, TRAIN: {:,} ({:.2f}%), DEV: {:,} ({:.2f}%), TEST: {:,} ({:.2f}%)\n".format(
	num_reviews, len(train_interactions), train_pct, len(dev_interactions), dev_pct, len(test_interactions), test_pct))

train_users, train_items = get_users_items(train_interactions)
append_to_file(output_log, "[FINAL Stats][TRAIN] Users: {:,}, Items: {:,}, Ratings: {:,}".format(
	len(train_users), len(train_items), len(train_interactions) ))

dev_users, dev_items = get_users_items(dev_interactions)
append_to_file(output_log, "[FINAL Stats][DEV]   Users: {:,}, Items: {:,}, Ratings: {:,}".format(
	len(dev_users), len(dev_items), len(dev_interactions) ))

test_users, test_items = get_users_items(test_interactions)
append_to_file(output_log, "[FINAL Stats][TEST]  Users: {:,}, Items: {:,}, Ratings: {:,}\n\n".format(
	len(test_users), len(test_items), len(test_interactions) ))



# Here, we construct the user & item documents
users_reviews = defaultdict(list)
items_reviews = defaultdict(list)

append_to_file(output_log, "\nConsolidating user/item reviews from TRAINING set")
for interaction in tqdm(train_interactions, desc = "Consolidating user/item reviews from TRAINING set"):

	user = interaction[0]
	item = interaction[1]

	text = interaction[4]

	users_reviews[user].append(text)
	items_reviews[item].append(text)

users_doc = defaultdict(list)
items_doc = defaultdict(list)

append_to_file(output_log, "\nCreating user docs from TRAINING set")
for user, reviews in users_reviews.items():

	random.shuffle(reviews)
	for review in reviews:

		currUserDocLen = len(users_doc[user])
		if(currUserDocLen < MAX_DOC_LEN):
			users_doc[user].extend( review[:(MAX_DOC_LEN - currUserDocLen)] )
		else:
			break

append_to_file(output_log, "Creating item docs from TRAINING set")
for item, reviews in items_reviews.items():

	random.shuffle(reviews)
	for review in reviews:

		currItemDocLen = len(items_doc[item])
		if(currItemDocLen < MAX_DOC_LEN):
			items_doc[item].extend( review[:(MAX_DOC_LEN - currItemDocLen)] )
		else:
			break

# Force garbage collection
del users_reviews
del items_reviews
gc.collect()


# Just checking the document length
minUserDocLen = MAX_DOC_LEN
minItemDocLen = MAX_DOC_LEN

for user, user_doc in users_doc.items():
	currUserDocLen = len(user_doc)
	minUserDocLen = min(minUserDocLen, currUserDocLen)

for item, item_doc in items_doc.items():
	currItemDocLen = len(item_doc)
	minItemDocLen = min(minItemDocLen, currItemDocLen)

append_to_file(output_log, "\nMinimum User Doc Len: {}, Minimum Item Doc Len: {}".format(minUserDocLen, minItemDocLen))



# Get vocabulary based on the user & item documents
words = []
for user, user_doc in users_doc.items():
	words += [word for word in user_doc]
for item, item_doc in items_doc.items():
	words += [word for word in item_doc]

# Get unique words to form the vocabulary
append_to_file(output_log, "\nOriginal number of words (based on USER & ITEM documents constructed from TRAINING set): {:,}".format( len(list(set(words))) ))
append_to_file(output_log, "For the vocabulary, we are only using the {:,} most frequent words".format( VOCAB_SIZE ))

# NOTE: Word order in "words" is relative to frequency, i.e. Most frequent, 2nd-most frequent, and so on..
words_count = Counter(words).most_common(VOCAB_SIZE)
words = [w for w, c in words_count]
append_to_file(output_log, "Current number of words: {:,}\n".format( len(words)) )

# Build word dictionary - PAD is 0, UNK (OOV) is 1
word_wid = {word: (wid + 2) for wid, word in enumerate(words)}
word_wid[PAD] = 0
word_wid[UNK] = 1

# Build user/item dictionary
user_uid = {user: uid for uid, user in enumerate(users_doc.keys())}
item_iid = {item: iid for iid, item in enumerate(items_doc.keys())}

# Convert each user/item to its correspoding index in the user/item dictionary
# Convert each word to its corresponding index in the word dictionary
print("For each user/item doc, converting words to wids using word_wid..\n")
uid_userDoc = {user_uid[user]: doc2id(userDoc, word_wid) for user, userDoc in users_doc.items()}
iid_itemDoc = {item_iid[item]: doc2id(itemDoc, word_wid) for item, itemDoc in items_doc.items()}

# Store the actual length of each user/item document (before padding)
uid_userDocLen = {uid: len(userDoc) for uid, userDoc in uid_userDoc.items()}
iid_itemDocLen = {iid: len(itemDoc) for iid, itemDoc in iid_itemDoc.items()}

# Pad the user/item documents to MAX_DOC_LEN
uid_userDoc = {uid: post_padding(userDoc, MAX_DOC_LEN, word_wid[PAD]) for uid, userDoc in uid_userDoc.items()}
iid_itemDoc = {iid: post_padding(itemDoc, MAX_DOC_LEN, word_wid[PAD]) for iid, itemDoc in iid_itemDoc.items()}

# Force garbage collection
gc.collect()



train = prepare_set(train_interactions, user_uid, item_iid, uid_userDocLen, iid_itemDocLen, "TRAINING", output_log)
dev = prepare_set(dev_interactions, user_uid, item_iid, uid_userDocLen, iid_itemDocLen, "DEV", output_log)
test = prepare_set(test_interactions, user_uid, item_iid, uid_userDocLen, iid_itemDocLen, "TESTING", output_log)


env = {

	# List of (word, frequency)
	'words_count': words_count,

	# Mapping from word to wid in the word dictionary
	'word_wid': word_wid,

	# Mapping from user to uid in the user dictionary (More for finding user based on uid)
	'user_uid': user_uid,

	# Mapping from item to iid in the item dictionary (More for finding item based on iid)
	'item_iid': item_iid,

	# Mapping from uid to the original userDocLen
	'uid_userDocLen': uid_userDocLen,

	# Mapping from iid to the original itemDocLen
	'iid_itemDocLen': iid_itemDocLen
}


# Some useful information
info = {

	'num_users': len(user_uid),
	'num_items': len(item_iid),
	'num_ratings': (len(train_interactions) + len(dev_interactions) + len(test_interactions)),
	'train_size': len(train_interactions),
	'dev_size': len(dev_interactions),
	'test_size': len(test_interactions)
}



append_to_file(output_log, "\nSaving all required files for \"{}\"..".format( CATEGORY ))

# Training set
with open(output_split_train, "wb+") as f:
	pickle.dump(train, f, protocol = 2)
append_to_file(output_log, "{:<21s} {}".format( "Training Set:", output_split_train ))

# Validation set
with open(output_split_dev, "wb+") as f:
	pickle.dump(dev, f, protocol = 2)
append_to_file(output_log, "{:<21s} {}".format( "Validation Set:", output_split_dev ))

# Testing set
with open(output_split_test, "wb+") as f:
	pickle.dump(test, f, protocol = 2)
append_to_file(output_log, "{:<21s} {}".format( "Test Set:", output_split_test ))

# env
with open(output_env, "wb+") as f:
	pickle.dump(env, f, protocol = 2)
append_to_file(output_log, "{:<21s} {}".format( "Environment:", output_env ))

# info
with open(output_info, "wb+") as f:
	pickle.dump(info, f, protocol = 2)
append_to_file(output_log, "{:<21s} {}".format( "Info:", output_info ))


# User Documents
append_to_file(output_log, "\nCreating numpy matrix for uid_userDoc..")
uid_userDoc_Matrix = createNumpyMatrix(0, len(uid_userDoc), uid_userDoc)
np.save(output_uid_userDoc, uid_userDoc_Matrix)
append_to_file(output_log, "{:<21s} {}".format( "User Document Matrix:", uid_userDoc_Matrix.shape ))
append_to_file(output_log, "{:<21s} {}".format( "User Document Matrix:", output_uid_userDoc ))


# Item Documents
append_to_file(output_log, "\nCreating numpy matrix for iid_itemDoc..")
iid_itemDoc_Matrix = createNumpyMatrix(0, len(iid_itemDoc), iid_itemDoc)
np.save(output_iid_itemDoc, iid_itemDoc_Matrix)
append_to_file(output_log, "{:<21s} {}".format( "Item Document Matrix:", iid_itemDoc_Matrix.shape ))
append_to_file(output_log, "{:<21s} {}".format( "Item Document Matrix:", output_iid_itemDoc ))


# Force garbage collection
del train
del dev
del test
del env
del info
del uid_userDoc_Matrix
del iid_itemDoc_Matrix
gc.collect()


append_to_file(output_log, "\nAll required files for \"{}\" successfully saved to '{}'".format( CATEGORY, CATEGORY_FOLDER ))



endTime = time.time()
durationInSecs = endTime - startTime
durationInMins = durationInSecs / 60
append_to_file(output_log, "\nPreprocessing for \"{}\" done after {:.2f} seconds ({:.2f} minutes)\n".format( CATEGORY, durationInSecs, durationInMins ))

# Force garbage collection
gc.collect()

print("\nDone!!\n")
exit()


