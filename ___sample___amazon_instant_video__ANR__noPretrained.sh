# Example script for ANR
# We repeat the process 5 times using different random seeds

# Train everything from scratch
python3 PyTorchTEST.py -d "amazon_instant_video" -m "ANR" -e 25 -p 1 -rs 1337 -gpu 0 -vb 1
python3 PyTorchTEST.py -d "amazon_instant_video" -m "ANR" -e 25 -p 1 -rs 1234 -gpu 0 -vb 1
python3 PyTorchTEST.py -d "amazon_instant_video" -m "ANR" -e 25 -p 1 -rs 5678 -gpu 0 -vb 1
python3 PyTorchTEST.py -d "amazon_instant_video" -m "ANR" -e 25 -p 1 -rs 1357 -gpu 0 -vb 1
python3 PyTorchTEST.py -d "amazon_instant_video" -m "ANR" -e 25 -p 1 -rs 2468 -gpu 0 -vb 1