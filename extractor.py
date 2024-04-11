"""
Author: Viet Nguyen
Date: 2022-08-28
"""

# %%

##### CLASS ORDER ###########
##### Backgrund: 0
##### Sclera: 1
##### Iris: 2
##### PUPIL: 3

import os

from vietlib.data.extractor import OpenEDSExtractor, OptimizedRITEyesNewPipelineExtractor

os.makedirs("data", exist_ok=True)

# image_size = (320, 240) # Small
# image_size = (400, 320) # large
image_size = (400, 640) # original

data_dir = "/data/OpenEDS/Semantic_Segmentation_Dataset" # Path to your downloaded dataset
target_path = "./data/open_eds_real.h5"
e = OpenEDSExtractor(data_dir, target_path, image_size=image_size, val_with_train=True, keep_original=True)
e.extract(resizing=False)


data_dir = "/data/new_rit_eyes/raw" # Path to your downloaded dataset
target_path = "./data/rit_eyes.h5"
e = OptimizedRITEyesNewPipelineExtractor(data_dir, target_path, image_size=image_size, val_with_train=True)
e.extract(resizing=False)


