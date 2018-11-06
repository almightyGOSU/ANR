# Example script for ANR
# We repeat the process 5 times using different random seeds

# If pretrained ARL weights are available, specify it using -ARL_path ...
# The saved ARNS model weights should be in ./__saved_models__/[dataset] - ARNS/[dataset]_ANRS_[random_seed].pth
# E.g. ./__saved_models__/amazon_instant_video - ARNS/amazon_instant_video_ANRS_1337.pth
python3 PyTorchTEST.py -d "amazon_instant_video" -m "ANR" -e 15 -p 1 -rs 1337 -gpu 0 -vb 1 -ARL_path "amazon_instant_video_ANRS_1337"
python3 PyTorchTEST.py -d "amazon_instant_video" -m "ANR" -e 15 -p 1 -rs 1234 -gpu 0 -vb 1 -ARL_path "amazon_instant_video_ANRS_1337"
python3 PyTorchTEST.py -d "amazon_instant_video" -m "ANR" -e 15 -p 1 -rs 5678 -gpu 0 -vb 1 -ARL_path "amazon_instant_video_ANRS_1337"
python3 PyTorchTEST.py -d "amazon_instant_video" -m "ANR" -e 15 -p 1 -rs 1357 -gpu 0 -vb 1 -ARL_path "amazon_instant_video_ANRS_1337"
python3 PyTorchTEST.py -d "amazon_instant_video" -m "ANR" -e 15 -p 1 -rs 2468 -gpu 0 -vb 1 -ARL_path "amazon_instant_video_ANRS_1337"