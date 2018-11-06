# Example script for ANRS (i.e. the 'Simplified Model' used for obtaining the pretrained weights of the ARL layer)
# Model Pretraining for ANR, i.e. the ARL layer
python3 PyTorchTEST.py -d "amazon_instant_video" -m "ANRS" -e 10 -p 1 -rs 1337 -gpu 0 -vb 1 -sm "amazon_instant_video_ANRS"