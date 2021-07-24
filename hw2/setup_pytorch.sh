#!bin/bash
conda update -n base conda
conda create -n pytorch-gpu ipykernel
source activate pytorch-gpu
conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

conda install -y numpy matplotlib pandas scikit-learn

conda install -c conda-forge seaborn pandas-gbq fsspec gcsfs 'google-cloud-bigquery[bqstorage,pandas]' google-api-python-client google-auth-httplib2 google-auth-oauthlib google-auth-oauthlib -y

pip install gpustat
# conda install tensorflow-gpu

# check if pytorch can find GPU
# import torch
# torch.cuda.current_device()
# # 0
# torch.cuda.get_device_name(0)
# # 'Tesla K80'