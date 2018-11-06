Hello!



Step 0. We asssume the following directory structure:

test/                                   # root folder

  GoogleNews-vectors-negative300.bin    # this is the file for pretrained word embeddings

  ANR/                                  # Basically, what you clone from github..

    __saved_models__/                   # This is where the pretrained ARL weights go to..

    datasets/                           # Datasets, this includes the downloaded json files for Amazon & Yelp
      amazon_instant_video/             # E.g. A folder for Amazon Instant Video (automatically created by the preprocessing code)

    experimental_results/               # This is where your results go to..

    model/                              # All model-related code (i.e. all the PyTorch stuff)

    preprocessing/                      # All preprocessing code (i.e. for generating the required files from downloaded json files)

    FILEPATHS.py                        # Names of files shared across all code
    PyTorchTEST.py                      # Basically main.py.. The model is trained and tested here (despite the weird filename)



Step 1. Preprocessing

  - NOTE: For this step, your current directory should be the 'preprocessing' folder..
  - E.g. test/ANR/preprocessing/ in the example directory structure!

  - refer to ___notes___preprocessing_part_1.txt
  - refer to ___notes___preprocessing_part_2.txt



Step 2. Running the model

  - NOTE: For this step, your current directory should be the 'ANR' folder.. i.e. test/ANR/ in the example directory structure!

  - If you want to train & test the model directly..
    - refer to ___sample___amazon_instant_video__ANR__noPretrained.sh

  - If you want to (1) pretrain the weights for ANR, i.e. the weights for the ARL layer,
  - and (2) train & test the model with these pretrained weights..
    - refer to ___sample___amazon_instant_video__ANR___SIMPLE.sh
    - refer to ___sample___amazon_instant_video__ANR.sh



[ Miscellaneous Information ]

Experiments were run on a Ubuntu server with version 14.04.5 LTS, conda 4.5.0, python 3.6.3, and pytorch 0.3.0.

Yelp dataset
- Latest version (Round 11) of the Yelp Dataset Challenge
- Obtained from: https://www.yelp.com/dataset/challenge

Amazon datasets
- Amazon Product Reviews, which has been organized into 24 individual product categories
- Obtained from: http://jmcauley.ucsd.edu/data/amazon/



[ Optional ]
An example using the Amazon Instant Video dataset:


(1) Download the json file and put it in the 'datasets' folder

  - e.g. test/ANR/datasets/amazon_instant_video.json


(2) Download the pretrained word2vec embeddings if you haven't done so..

  - If you are following the example directory structure, there is no need to change anything
  - If not, please edit this line "input_embeddings   = "../../GoogleNews-vectors-negative300.bin" in pretrained_vectors_simple.py


(3) Preprocessing Part 1

  - cd to 'preprocessing' folder

  - For example, run this:
    python3 preprocessing_simple.py -d amazon_instant_video -dev_test_in_train 1

  - change the -d argument for other datasets

  - there will be a new folder within 'datasets', e.g. test/ANR/datasets/amazon_instant_video/
  - there will be a total of 8 files inside


(4) Preprocessing Part 2

  - cd to 'preprocessing' folder

  - For example, run this:
    python3 pretrained_vectors_simple.py -d amazon_instant_video

  - change the -d argument for other datasets

  - there will be 2 new files added to the folder, e.g. test/ANR/datasets/amazon_instant_video/
  - there will be a total of 10 files inside


(5) Model Part 1 - Pretraining

  - cd to 'ANR' folder

  - For example, run this:
    python3 PyTorchTEST.py -d "amazon_instant_video" -m "ANRS" -e 10 -p 1 -rs 99 -gpu 5 -vb 1 -sm "amazon_instant_video_ANRS"

  - change the -d argument for other datasets
  - similarly, change -sm to save the pretrained weights to a different file

  - basically, we run the simplified model for 10 epochs to get pretrained weights for the ARL layer

  - the weights are saved to the '__saved_models__' folder
  - e.g. test/ANR/__saved_models__/amazon_instant_video - ANRS/amazon_instant_video_ANRS_1337.pth

  - model output (some information & results) are saved to the 'experimental_results' folder
  - e.g. test/ANR/experimental_results/amazon_instant_video - ANRS/2018-11-02-22-23-58-logs.txt
  - the results from this part of the model training are not very useful

  - NOTE: simplified model == ANRS, and the complete model == ANR


(6) Model Part 2 - Actual Model

  - cd to 'ANR' folder

  - For example, run this:
    python3 PyTorchTEST.py -d "amazon_instant_video" -m "ANR" -e 15 -p 1 -rs 99 -gpu 7 -vb 1 -ARL_path "amazon_instant_video_ANRS_1337"

  - change the -d argument for other datasets
  - similarly, change -ARL_path to load the pretrained weights from a different file

  - basically, we run the complete model for 15 epochs to obtain results
  - repeat with other random seeds by changing the -rs argument

  - model output (some information & results) are saved to the 'experimental_results' folder
  - e.g. test/ANR/experimental_results/amazon_instant_video - ANR/2018-11-04-15-57-00-logs.txt

  - what this file contains:
    - the input files, some information such as number of users, and number of items..
    - model size, what are the trainable parameters
    - for each epoch: the training loss, the dev MSE, and the test MSE, as well as time taken
    - at the end of the file, it shows the best dev MSE, when the best dev MSE was obtained, and the corresponding test MSE




[ Optional ]

- For running the model, everything starts from PyTorchTEST.py
  - It contains all the training and evaluation code

- The model code starts from ModelZoo.py
  - Here we create the model (and any review-based baseline models)

- Relevant ANR code can be found in:
  - (1) ANR.py
  - (2) ANR_AIE.py
  - (3) ANR_ARL.py
  - (4) ANR_RatingPred.py

- The simplified model (basically, ANR without AIE) can be found in:
  - (1) ANRS_RatingPred.py

- Remaining files are just helper classes

- Keys Arguments for PyTorchTEST.py
  -d:     dataset, e.g. amazon_instant_video, musical_instruments, etc..
  -m:     model, e.g. ANRS or ANR
  -e:     number of epochs
  -K:     number of aspects
  -h1:    dimensionality of aspect-level user & item representations
  -h2:    size of hidden layers in Aspect Importance Estimation



