"""
Author: Viet Nguyen
Date: 2022-11-27
"""

import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
import tensorflow
from tqdm.notebook import tqdm
import warnings
import pandas as pd

import sys

import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import h5py
import deepdish
import argparse

from vietlib.data import CVDataset, PairedImageDatasetWithLabel
from vietlib.data.transforms import MaskToTensor
from vietlib.model.cgan import Discriminator, GeneratorResNet
from vietlib.model.loss import StructureRetainLoss
from vietlib.model.segment import SobelMultiChannel
from vietlib.model.cnn import ResidualBlock
from vietlib.model.siamese import SiameseNet2D
from vietlib.train import Trainer, save_experiment
from vietlib.train.cgan import CGANTrainer_v2
from vietlib.utils import check_vscode_interactive, get_nparams, mIoU_v2
from vietlib.data.transforms import transform_sequence
from vietlib.utils.experiment import PROFILE_NAME, CVExperiment, Experiment, Experiment_v2, MLExperiment
from vietlib.utils.image import one_hot_label, show_img
from vietlib.data.extractor import Extractor
from vietlib.model.segment import DenseNet2D


transforms_ = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.5], [0.5])
])

# Default config
parser = argparse.ArgumentParser()

parser.add_argument("-l", "--log_root_folder", default="logs/dda_pipeline/cgan_export")
parser.add_argument("-e", "--experiment_name", default="test_1")

parser.add_argument("--target_path", default="./cycled_synth_filter.h5")
parser.add_argument("--model_num", default="5", help="can be -1 or any other postfixes of saved models")
parser.add_argument("--use_GPU", default=True)
parser.add_argument("--model_directory", default="logs/dda_pipeline/srcgan/srcgan_60_01_01/checkpoints", help="model for cgan weights")
parser.add_argument("--source_domain", default="/data/rit_eyes.h5")

parser.add_argument("--image_height", default=640, type=int)
parser.add_argument("--image_width", default=400, type=int)
parser.add_argument("--n_classes", default=4, type=int)
parser.add_argument("--channel", default=1, type=int)
parser.add_argument("--n_residual_blocks", default=8, type=int)
parser.add_argument("--use_segment", default=False)
parser.add_argument("--valid_step", default=50, type=int)
parser.add_argument("--batch_size", default=1, type=int)

# sia config
parser.add_argument("--sia_filter", action='store_true', default=False)
parser.add_argument("--target_domain", default="/data/open_eds_real.h5")
parser.add_argument("--sia_weights_path", default="logs/dda_pipeline/siamese/sia_4/checkpoints/sia_4.pth")
parser.add_argument("--closed_threshold", default=0.05, type=float)

# ritnet config
parser.add_argument("--ritnet_weights_path", default="logs/dda_pipeline/pop/ritnet_7_raw/fold_4/checkpoints/best_model.pth")
parser.add_argument("--starting_hidden_neuron", default=32, type=int)

# Whether to export the dataset as h5 file or not
parser.add_argument("--export_h5", default=False, action='store_true')

# parse args
if check_vscode_interactive():
  args = parser.parse_args([])
else:
  args = parser.parse_args()

# Bring to local var
target_path = args.target_path
model_num = args.model_num
threshold = float(args.closed_threshold)
sia_filter = args.sia_filter

E = Experiment_v2(log_root_folder=args.log_root_folder, experiment_name=args.experiment_name)
E.model_directory = args.model_directory

E.on_experiment_start(args)

if args.use_GPU and torch.cuda.is_available():
  device=torch.device("cuda")
else:
  device=torch.device("cpu")

# Siamese network initialization
val_real_data = CVDataset(args.target_domain, "validation", transforms_, data_point_limit=500)
val_real_loader = DataLoader(val_real_data, batch_size=args.batch_size, drop_last=True)

# Model
sia_net = SiameseNet2D(
  (args.channel, args.image_height, args.image_width),
  n_classes=2
)

# Construct model path
# sia_model_path = os.path.join(sia_E.log_directory, "checkpoints", sia_E.config.experiment_name + ".pth")
sia_model_path = args.sia_weights_path
print(f"sia_net: n_params: {get_nparams(sia_net)} from {sia_model_path}")
# Path for saving training results in numpy formats
sia_net.load_state_dict(torch.load(sia_model_path))
sia_net = sia_net.to(device)

# Build mean example representation vector
example_data = h5py.File(args.target_domain, 'r')["val"]
n_val = example_data["image"].__len__()

real_vec_list = []

with torch.no_grad():
  for i, batch in enumerate(val_real_loader):
    img, _, _, _ = batch

    if i % 100 == 0:
      print(f"Process {i} out of {len(val_real_loader)}")
    example_real_vec = sia_net.forward_once(img.to(device))
    real_vec_list.append(example_real_vec)

repr_vector = torch.concat(real_vec_list, dim=0).mean(dim=0, keepdim=True)

print(f"repr_vector: {repr_vector.cpu().detach().numpy()}")


img_height = args.image_height
img_width = args.image_width
channels = args.channel
input_shape = (channels, img_height, img_width) 

batch_size = args.batch_size # size of the batches
b1 = 0.5 # adam : decay of first order momentum of gradient
b2 = 0.999 # adam : decay of first order momentum of gradient

n_residual_blocks = args.n_residual_blocks # suggested default 11, number of residual blocks in generator

G_AB = GeneratorResNet(input_shape, args.n_classes, n_residual_blocks)
G_BA = GeneratorResNet(input_shape, args.n_classes, n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

sobel_img = SobelMultiChannel(1, normalize=True).to(device)

def load_model(ckpt_path=None):

  if os.path.exists(ckpt_path):
    ckpt_data = torch.load(ckpt_path)
    # self.current_epoch = ckpt_data["epoch"] + 1
    G_AB.load_state_dict(ckpt_data["G_AB"])
    G_BA.load_state_dict(ckpt_data["G_BA"])
    D_A.load_state_dict(ckpt_data["D_A"])
    D_B.load_state_dict(ckpt_data["D_B"])
  else:
    print("No weights loaded!")

  G_AB.to(device)
  G_BA.to(device)
  D_A.to(device)
  D_B.to(device)

# Load model
if model_num != "-1":
  model_path = os.path.join(E.model_directory, f"ckpt_epoch_{model_num}.pth")
  print(f"load model {model_path}")
else:
  model_path = os.path.join(E.model_directory, "ckpt.pth")
  print(f"load model {model_path}")

load_model(model_path)

######## Extracting

import cv2
def export_out_of_distribution_image(img: np.ndarray, image_dir: str, img_id: str) -> None:
  cv2.imwrite(os.path.join(image_dir, f"{img_id}.png"), img)

# Actual writing data
chunk_size = 500
train_data = h5py.File(args.source_domain, 'r')["train"]
val_data = h5py.File(args.source_domain, 'r')["val"]
writing_data = Extractor.initialize_data()


img_shape = (args.image_height, args.image_width)

with h5py.File(target_path, 'w') as f:
  # Initialize a resizable dataset to hold the output
  # maxshape = (None,) + chunk.shape[1:]
  train = f.create_group("train")
  val = f.create_group("val")


  image = train.create_dataset('image', shape=(chunk_size, *img_shape), maxshape=(None, *img_shape),
                            chunks=(chunk_size, *img_shape), dtype=np.uint8)

  resolution = train.create_dataset('resolution', shape=(chunk_size, 2), maxshape=(None, 2),
                            chunks=(chunk_size, 2), dtype=np.uint8)

  label = train.create_dataset('label', shape=(chunk_size, *img_shape), maxshape=(None, *img_shape),
                            chunks=(chunk_size, *img_shape), dtype=np.uint8)
  row_count = 0

  i = 0
  count = 0
  while i < len(train_data["image"]):
    if count % chunk_size == 0:
      print(f"batch {i}")
      if count != 0:
        # stack
        try:
          writing_data["train"]["image"] = np.stack(writing_data["train"]["image"], axis=0)
          writing_data["train"]["resolution"] = np.stack(writing_data["train"]["resolution"], axis=0)
          # writing_data["train"]["path"] = np.stack(writing_data["train"]["path"], axis=0)
          writing_data["train"]["label"] = np.stack(writing_data["train"]["label"], axis=0)
          # Append image
          image.resize(row_count + writing_data["train"]["image"].shape[0], axis=0)
          image[row_count:] = writing_data["train"]["image"]
          resolution.resize(row_count + writing_data["train"]["resolution"].shape[0], axis=0)
          resolution[row_count:] = writing_data["train"]["resolution"]
          # path.resize(row_count + writing_data["train"]["path"].shape[0], axis=0)
          # path[row_count:] = writing_data["train"]["path"]
          label.resize(row_count + writing_data["train"]["label"].shape[0], axis=0)
          label[row_count:] = writing_data["train"]["label"]
          row_count += writing_data["train"]["image"].shape[0]
          writing_data = Extractor.initialize_data()
        except: # ValueError: need at least one array to stack
          pass
        
    img = train_data["image"][i]
    img = transforms_(img).unsqueeze(0).to(device)
    lbl = train_data["label"][i]
    lbl = MaskToTensor()(lbl).unsqueeze(0)
    lbl = one_hot_label(lbl, n_classes=args.n_classes).to(device)
    out = G_AB(img, lbl)
    
    if sia_filter:
      with torch.no_grad():
        # Siamese net filtering
        o0 = sia_net.forward_once(out)
        d = torch.nn.functional.pairwise_distance(o0, repr_vector).squeeze().cpu().item()
        if d >= threshold:
          i += 1
          continue
    
    # print("s")
    out = ((out.squeeze().cpu().detach() / 2 + 0.5) * 255).type(torch.uint8).numpy()
    writing_data["train"]["image"].append(out)
    writing_data["train"]["resolution"].append(train_data["resolution"][i])
    # writing_data["train"]["path"].append(train_data["path"][i])
    writing_data["train"]["label"].append(train_data["label"][i])

    i += 1
    count += 1

  # Append the last chunk
  # stack
  try:
    writing_data["train"]["image"] = np.stack(writing_data["train"]["image"], axis=0)
    writing_data["train"]["resolution"] = np.stack(writing_data["train"]["resolution"], axis=0)
    writing_data["train"]["label"] = np.stack(writing_data["train"]["label"], axis=0)
    # Append image
    image.resize(row_count + writing_data["train"]["image"].shape[0], axis=0)
    image[row_count:] = writing_data["train"]["image"]
    resolution.resize(row_count + writing_data["train"]["resolution"].shape[0], axis=0)
    resolution[row_count:] = writing_data["train"]["resolution"]
    label.resize(row_count + writing_data["train"]["label"].shape[0], axis=0)
    label[row_count:] = writing_data["train"]["label"]
    row_count += writing_data["train"]["image"].shape[0]
  except:
    pass


  writing_data = Extractor.initialize_data()
  row_count = 0

  # Process for val
  image = val.create_dataset('image', shape=(chunk_size, *img_shape), maxshape=(None, *img_shape),
                            chunks=(chunk_size, *img_shape), dtype=np.uint8)

  resolution = val.create_dataset('resolution', shape=(chunk_size, 2), maxshape=(None, 2),
                            chunks=(chunk_size, 2), dtype=np.uint8)

  label = val.create_dataset('label', shape=(chunk_size, *img_shape), maxshape=(None, *img_shape),
                            chunks=(chunk_size, *img_shape), dtype=np.uint8)

  i = 0
  count = 0
  while i < len(val_data["image"]):
    if count % chunk_size == 0:
      print(f"batch {i}")
      if count != 0:
        try: 
          writing_data["val"]["image"] = np.stack(writing_data["val"]["image"], axis=0)
          writing_data["val"]["resolution"] = np.stack(writing_data["val"]["resolution"], axis=0)
          writing_data["val"]["label"] = np.stack(writing_data["val"]["label"], axis=0)

          # Append image
          image.resize(row_count + writing_data["val"]["image"].shape[0], axis=0)
          image[row_count:] = writing_data["val"]["image"]
          resolution.resize(row_count + writing_data["val"]["resolution"].shape[0], axis=0)
          resolution[row_count:] = writing_data["val"]["resolution"]
          label.resize(row_count + writing_data["val"]["label"].shape[0], axis=0)
          label[row_count:] = writing_data["val"]["label"]
          row_count += writing_data["val"]["image"].shape[0]
          writing_data = Extractor.initialize_data()
        except: # ValueError: need at least one array to stack
          pass

    img = val_data["image"][i]
    img = transforms_(img).unsqueeze(0).to(device)
    lbl = val_data["label"][i]
    lbl = MaskToTensor()(lbl).unsqueeze(0)
    lbl = one_hot_label(lbl, n_classes=args.n_classes).to(device)
    out = G_AB(img, lbl)

    if sia_filter:
      with torch.no_grad():
        # Siamese net filtering
        o0 = sia_net.forward_once(out)
        d = torch.nn.functional.pairwise_distance(o0, repr_vector).squeeze().cpu().item()
        
        if d >= threshold:
          i += 1
          continue

    out = ((out.squeeze().cpu().detach() / 2 + 0.5) * 255).type(torch.uint8).numpy()
    writing_data["val"]["image"].append(out)
    writing_data["val"]["resolution"].append(val_data["resolution"][i])
    writing_data["val"]["label"].append(val_data["label"][i])

    i += 1
    count += 1
  
  try:
    writing_data["val"]["image"] = np.stack(writing_data["val"]["image"], axis=0)
    writing_data["val"]["resolution"] = np.stack(writing_data["val"]["resolution"], axis=0)
    writing_data["val"]["label"] = np.stack(writing_data["val"]["label"], axis=0)

    # Append image
    image.resize(row_count + writing_data["val"]["image"].shape[0], axis=0)
    image[row_count:] = writing_data["val"]["image"]
    resolution.resize(row_count + writing_data["val"]["resolution"].shape[0], axis=0)
    resolution[row_count:] = writing_data["val"]["resolution"]
    label.resize(row_count + writing_data["val"]["label"].shape[0], axis=0)
    label[row_count:] = writing_data["val"]["label"]
    row_count += writing_data["val"]["image"].shape[0]
    writing_data = Extractor.initialize_data()
  except:
    pass


######################## Export vectors and RIT Inference ########################
###################### RITnet Inference and get Mean IOU #####################

# Fake
val_synth_dataset = CVDataset(
  data_path=args.source_domain,
  split="validation", 
  transform=transform_sequence
)
val_synth_loader = DataLoader(
  val_synth_dataset,
  batch_size=E.config.batch_size,
  drop_last=True
)

# Distance of cycled to real
val_cycled_dataset = CVDataset(
  data_path=args.target_path, # the dataset_path for the new generated data file
  split="validation", 
  transform=transform_sequence,
  # shuffle=True
)
val_cycled_loader = DataLoader(
  val_cycled_dataset,
  batch_size=1,
  drop_last=True
)

# Ritnet initialization
ritnet = DenseNet2D(channel_size=E.config.starting_hidden_neuron, dropout=True,prob=0.2)
ritnet = ritnet.to(device)
ritnet.load_state_dict(torch.load(args.ritnet_weights_path))
ritnet.eval()

# Stats
vector_repr = np.zeros((0, 2)) # shape (N, 2)
n_not_closed = 0
d_to_real = []
ious = []

for i, batch in enumerate(val_cycled_loader):
  if i % 100 == 0:
    print(f"batch {i}")
  with torch.no_grad():
    img, label, _, _ = batch
    out = sia_net.forward_once(img.to(device))
    seg_map = ritnet(img.to(device))

    # Distance to the real
    d = torch.nn.functional.pairwise_distance(out, repr_vector).squeeze().cpu().item()
    if d >= threshold:
      n_not_closed += 1
  
  vector_repr = np.concatenate((vector_repr, out.cpu().numpy()), axis=0)
  d_to_real.append(d)

  mean_iou: float = mIoU_v2(seg_map.cpu(), label.cpu(), n_unique_labels=4)
  ious.append(mean_iou)

# Export csv vectors.
vec_df = pd.DataFrame(data = vector_repr, columns = ['x1', 'x2'])

# m_iou = np.mean(ious)
# Append to df
miou_df = pd.DataFrame.from_dict({
  "d2r": d_to_real,
  "miou": ious,
})

final_df = pd.concat((vec_df, miou_df), axis=1)
final_df.to_csv(os.path.join(E.log_directory, "stats.csv"), index=False)

sum_df = pd.DataFrame.from_dict({"mmiou": [np.mean(ious)], "notclose": [n_not_closed], "md2r": [np.mean(d_to_real)]})
sum_df.to_csv(os.path.join(E.log_directory, "sum_stats.csv"), index=False)

# Remove h5 file after processing to efficiently manage storage
if not args.export_h5:
  os.remove(args.target_path)

