#!/bin/bash -l

# Create and activate environment
conda create -n py38 pip python=3.8 -y
conda activate py38

# Installing pytorch
# this will install all the torch reporsitories including numpy
# since torch is backward compatible, we don't need to keep track of the version
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other required libraries
pip install tqdm==4.66.2 deepdish==0.3.7 tensorflow-cpu==2.13.0 \
  opencv-python==4.9.0.80 matplotlib==3.7.5 torchmetrics==0.11.1 \
  scikit-learn==1.1.2 munch==4.0.0 torchsummary==1.5.1 pandas==2.0.3
