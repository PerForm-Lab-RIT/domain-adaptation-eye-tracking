import collections
import gc
import numpy as np
import torch
from torch.utils.data import Dataset 
import os
from PIL import Image
from torchvision import transforms
import cv2
import random
import os.path as osp
import h5py
import matplotlib.pyplot as plt
import torchvision

from ..utils.image import get_eye_horizontal_window

from ..utils import free_memories, one_hot2dist, preprocess_label_sparse_np
import copy

from typing import Tuple, Dict, List
from .transforms import Starburst_augment, Line_augment, Gaussian_blur, Translation, RandomHorizontalFlip, Starburst_augment_raw_open_eds
from ..utils.experiment import Experiment

from ..data.transforms import MaskToTensor

from .transforms import transform_sequence
from ..utils.image import show_img

class OpenEDSRawDataset(Dataset):
    def __init__(self, filepath, split='train',transform=None):
        self.transform = transform
        self.filepath= osp.join(filepath,split)
        self.split = split
        listall = []
        
        for file in os.listdir(osp.join(self.filepath,'images')):   
            if file.endswith(".png"):
               listall.append(file.strip(".png"))
        self.list_files=listall
        
        #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS 
        #local Contrast limited adaptive histogram equalization algorithm
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        imagepath = osp.join(self.filepath,'images',self.list_files[idx]+'.png')
        pilimg = Image.open(imagepath).convert("L")
        H, W = pilimg.width , pilimg.height
       
        #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS 
        #Fixed gamma value for      
        table = 255.0*(np.linspace(0, 1, 256)**0.8)
        pilimg = cv2.LUT(np.array(pilimg), table)

        labelpath = osp.join(self.filepath,'labels',self.list_files[idx]+'.npy')
        label = np.load(labelpath)    
        label = np.resize(label,(W,H))
        label = Image.fromarray(label)     
               
        if self.transform is not None:
            if self.split == 'train':
                if random.random() < 0.2: 
                    pilimg = Starburst_augment_raw_open_eds()(np.array(pilimg))  
                if random.random() < 0.2: 
                    pilimg = Line_augment()(np.array(pilimg))    
                if random.random() < 0.2:
                    pilimg = Gaussian_blur()(np.array(pilimg))   
                if random.random() < 0.4:
                    pilimg, label = Translation()(np.array(pilimg),np.array(label))
                
        img = self.clahe.apply(np.array(np.uint8(pilimg)))    
        img = Image.fromarray(img)      
            
        if self.transform is not None:
            if self.split == 'train':
                img, label = RandomHorizontalFlip()(img,label)
            img = self.transform(img)    

        ## This is for boundary aware cross entropy calculation
        spatialWeights = cv2.Canny(np.array(label),0,3)/255
        spatialWeights=cv2.dilate(spatialWeights,(3,3),iterations = 1)*20
        spatialWeights = torch.tensor(np.asarray(spatialWeights))

        ##This is the implementation for the surface loss
        # Distance map for each class
        distMap = []
        for i in range(0, 4):
            distMap.append(one_hot2dist(np.array(label)==i))
        distMap = np.stack(distMap, 0)           
        distMap = np.float32(distMap)
        distMap = torch.tensor(distMap)

        label = MaskToTensor()(label)

        return img, label,spatialWeights, distMap

class OpenEDSDataset(torch.utils.data.Dataset):
  def __init__(self, data_folder: str, image_size: Tuple=(160, 100), split: str="train", transform=None, debugging: bool=False) -> None:
    self.transform = transform
    self.data_folder = data_folder
    self.split = split

    self.image_size = image_size ### (height, width)
    
    if not debugging:
      train_folder = [str(i) for i in range(1, 18, 1)]
      val_folder = [str(i) for i in range(18, 22, 1)]
      test_folder = [str(i) for i in range(22, 25, 1)]
    else:
      train_folder = ["1", "2"]
      val_folder = ["3"]
      test_folder = ["4"]

    if split == "train":
      self.folders = train_folder # The list of predefined training folders [1...17]
    elif split == "validation":
      self.folders = val_folder
    elif split == "test":
      self.folders = test_folder
    else:
        raise Exception(f"split should only be `train`, `test`, or `validation`")

    self.file_pointer = 0
    self.folder_pointer = 0

    # dictionary of file path
    file_path_dict = {}
    file_name_dict = {}
    dict_ptr = 0

    for i in range(len(self.folders)):
      _path = osp.join(self.data_folder, self.folders[i], "synthetic")
      _path_2 = osp.join(self.data_folder, self.folders[i], "mask-withskin")
      for image_name in os.listdir(_path):
        _img_path = osp.join(_path, image_name)
        _label_path = osp.join(_path_2, image_name)
        _img_name, _ = os.path.splitext(image_name)
        file_path_dict[dict_ptr] = (_img_path, _label_path)
        file_name_dict[dict_ptr] = f"{self.folders[i]}_{_img_name}"
        dict_ptr += 1

    self.file_path_dict = file_path_dict
    self.file_name_dict = file_name_dict
    self.dict_ptr = dict_ptr

    #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS 
    #local Contrast limited adaptive histogram equalization algorithm
    self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

  def __len__(self):
    return self.dict_ptr

  def __getitem__(self, idx):
    # Dealing with corrupted data file. https://stackoverflow.com/a/71493112
    try:
      # Read image
      image_path = self.file_path_dict[idx][0]
      pil_img = Image.open(image_path).convert("L") # Convert to grayscale, shape (H, W)
      pil_img = pil_img.resize(self.image_size[::-1]) # Since resizing in PIL take (width, height, instead of height, width)
      # Read label
      if self.split != "test":
        label_path: str = self.file_path_dict[idx][1]
        label_pil_img = Image.open(label_path)
        label_pil_img = label_pil_img.resize(self.image_size[::-1])
        label_img: np.ndarray = np.asarray(label_pil_img) # (H, W, 3)
    except:
      # Dealing with corrupted data file. https://stackoverflow.com/a/71493112
      idx = np.random.randint(0, len(self)-1)
      sample = self[idx]
      return sample

    (width, height) = pil_img.size

    if self.split != "test":
      
      assert label_img.shape[:2] == (height, width), "Unmatched label and image size"
      label: np.ndarray = preprocess_label_sparse_np(label_img) # (H, W)
      tensor_label = MaskToTensor()(label) # Pytorch tensor (H, W)
    
    # After reading, process
    # PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS
    # Fixed gamma value
    table = 255.0*(np.linspace(0, 1, 256)**0.8)
    pil_img_after = cv2.LUT(np.array(pil_img), table)
    img = self.clahe.apply(np.array(np.uint8(pil_img_after)))
    img = Image.fromarray(img)
    
    if self.split != 'test':
        
      ## This is for boundary aware cross entropy calculation
      spatial_weights = cv2.Canny(label_img, 0, 3) / 255
      spatial_weights = cv2.dilate(spatial_weights, (3, 3), iterations = 1) * 20
      
      # This is the implementation for the surface loss
      # Distance map for each class
      distance_map = []
      for i in range(0, 4):
        distance_map.append(one_hot2dist(label == i))
      distance_map = np.stack(distance_map, 0)
      
    if self.transform is not None:
      img = self.transform(img)
    
    if self.split == "test":
      return img, self.file_name_dict[idx]

    return img, tensor_label, spatial_weights, distance_map, self.file_name_dict[idx]


class DannDataset(torch.utils.data.Dataset):
  def __init__(self, experiment: Experiment, split: str="train", transform=None, debugging: bool=False) -> None:
    self.E = experiment
    self.transform = transform
    self.source_domain_folder = self.E.config.source_domain
    self.target_domain_folder = self.E.config.target_domain
    self.split = split
    

    self.image_size = self.E.config.training_image_size ### (height, width)
    
    # Since open-eds and s-general dataset has the same folder structure, we just need to the folder numbers.
    if not debugging:
      train_folder = [str(i) for i in range(1, 18, 1)]
      val_folder = [str(i) for i in range(18, 22, 1)]
      test_folder = [str(i) for i in range(22, 25, 1)]
    else:
      train_folder = ["1", "2"]
      val_folder = ["3"]
      test_folder = ["4"]

    if split == "train":
      self.folders = train_folder # The list of predefined training folders [1...17]
    elif split == "validation":
      self.folders = val_folder
    elif split == "test":
      self.folders = test_folder
    else:
        raise Exception(f"split should only be `train`, `test`, or `validation`")

    self.file_pointer = 0
    self.folder_pointer = 0

    # dictionary of file path
    file_path_dict = {}
    file_name_dict = {}
    dict_ptr = 0

    len_image_extension = len(self.E.config.image_extension)

    for i in range(len(self.folders)):
      _path = osp.join(self.source_domain_folder, self.folders[i], "synthetic")
      _path_2 = osp.join(self.source_domain_folder, self.folders[i], "mask-withskin")
      for image_name in os.listdir(_path):
        _img_path = osp.join(_path, image_name)
        _label_path = osp.join(_path_2, image_name)
        file_path_dict[dict_ptr] = (_img_path, _label_path)
        file_name_dict[dict_ptr] = f"{self.folders[i]}_{image_name[:-len_image_extension]}"
        dict_ptr += 1

    self.source_domain_length = dict_ptr

    for i in range(len(self.folders)):
      _path = osp.join(self.target_domain_folder, self.folders[i], "synthetic")
      _path_2 = osp.join(self.target_domain_folder, self.folders[i], "mask-withskin")
      for image_name in os.listdir(_path):
        _img_path = osp.join(_path, image_name)
        _label_path = osp.join(_path_2, image_name)
        file_path_dict[dict_ptr] = (_img_path, _label_path)
        file_name_dict[dict_ptr] = f"{self.folders[i]}_{image_name[:-len_image_extension]}"
        dict_ptr += 1

    self.file_path_dict = file_path_dict
    self.file_name_dict = file_name_dict
    self.dict_ptr = dict_ptr

    #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS 
    #local Contrast limited adaptive histogram equalization algorithm
    self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

  def __len__(self):
    return self.dict_ptr

  def _compute_crop_size_for_openeds(self, height, width):
    return (height * self.E.config.h_over_w_ratio, width)

  def _perform_random_crop(self, pil_img: Image):
    """ Naive way of random crop

    Args:
        pil_img (Image): _description_

    Returns:
        Image: _description_
        Tuple[int]
    """
    w, h = pil_img.size
    starting_pixel = random.randint(80, 120) # Choose a starting pixel from 80, 120
    crop_info = (0, starting_pixel, w, starting_pixel + int(self.E.h_over_w_ratio * w)) # Pil doc: (left, top, right, bottom)
    crop_img = pil_img.crop(crop_info)
    return crop_img, crop_info

  def _perform_crop_with_crop_info(self, pil_img: Image, crop_info: Tuple[int]) -> Image:
    return pil_img.crop(crop_info)

  def _is_target_domain(self, idx):
    # Source domain class is 0
    # Target domain class is 1
    return idx >= self.source_domain_length

  def __getitem__(self, idx):
    # Dealing with corrupted data file. https://stackoverflow.com/a/71493112
    try:
      # Read image
      image_path = self.file_path_dict[idx][0]
      pil_img = Image.open(image_path).convert("L") # Convert to grayscale, shape (H, W)
      
      # If this is the image from the openeds domain (the source domain), the perform random crop
      if not self._is_target_domain(idx):
        pil_img, crop_info = self._perform_random_crop(pil_img)
      
      pil_img = pil_img.resize(self.image_size[::-1]) # Since resizing in PIL take (width, height, instead of height, width)
      
      # print(pil_img.size)
      
      # Read label
      if self.split != "test":
        label_path: str = self.file_path_dict[idx][1]
        label_pil_img = Image.open(label_path)
        
        # Perform cropping the label according to the cropinfo on corpped input image
        if not self._is_target_domain(idx):
          label_pil_img = self._perform_crop_with_crop_info(label_pil_img, crop_info)

        label_pil_img = label_pil_img.resize(self.image_size[::-1])

        # print(label_pil_img.size)
        label_img: np.ndarray = np.asarray(label_pil_img) # (H, W, 3)
    except:
      # Dealing with corrupted data file. https://stackoverflow.com/a/71493112
      idx = np.random.randint(0, len(self)-1)
      sample = self[idx]
      return sample

    (width, height) = pil_img.size

    if self.split != "test":
      
      assert label_img.shape[:2] == (height, width), "Unmatched label and image size"
      label: np.ndarray = preprocess_label_sparse_np(label_img) # (H, W)
      tensor_label = MaskToTensor()(label) # Pytorch tensor (H, W)
    
    # After reading, process
    # PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS
    # Fixed gamma value
    table = 255.0*(np.linspace(0, 1, 256)**0.8)
    pil_img_after = cv2.LUT(np.array(pil_img), table)
    img = self.clahe.apply(np.array(np.uint8(pil_img_after)))
    img = Image.fromarray(img)
    
    if self.split != 'test':
        
      ## This is for boundary aware cross entropy calculation
      spatial_weights = cv2.Canny(label_img, 0, 3) / 255
      spatial_weights = cv2.dilate(spatial_weights, (3, 3), iterations = 1) * 20
      
      # This is the implementation for the surface loss
      # Distance map for each class
      distance_map = []
      for i in range(0, 4):
        distance_map.append(one_hot2dist(label == i))
      distance_map = np.stack(distance_map, 0)
      
    if self.transform is not None:
      img = self.transform(img)
    
    if self.split == "test":
      return img, self.file_name_dict[idx], self._is_source_domain(idx)

    return img, tensor_label, spatial_weights, distance_map, self.file_name_dict[idx], torch.tensor(self._is_target_domain(idx)).unsqueeze(-1).float()

class DannDataset2(torch.utils.data.Dataset):
  def __init__(self, experiment: Experiment, split: str="train", transform=None) -> None:
    self.E = experiment
    self.transform = transform
    # Source is real eye imagery: openeds2019
    self.source_domain_folder = self.E.config.source_domain
    # Target is synthetic openeds
    self.target_domain_folder = self.E.config.target_domain
    self.split = split
    
    self.image_size = self.E.config.training_image_size ### (height, width)
    
    # Since open-eds and s-general dataset has the same folder structure, we just need to the folder numbers.

    train_folder = [str(i) for i in range(1, 18, 1)]
    val_folder = [str(i) for i in range(18, 22, 1)]
    test_folder = [str(i) for i in range(22, 25, 1)]
    
    if split == "train":
      self.folders = train_folder # The list of predefined training folders [1...17]
    elif split == "validation":
      self.folders = val_folder
    elif split == "test":
      self.folders = test_folder
    else:
        raise Exception(f"split should only be `train`, `test`, or `validation`")

    self.file_pointer = 0
    self.folder_pointer = 0

    # dictionary of file path
    file_path_dict = {}
    file_name_dict = {}
    dict_ptr = 0

    len_image_extension = len(self.E.config.image_extension)

   
    if split == "train":
      _path = osp.join(self.source_domain_folder, "train", "images")
      _path2 = osp.join(self.source_domain_folder, "train", "labels")
      for fname in os.listdir(_path):
        _img_path = os.path.join(_path, fname)
        _label_path = os.path.join(_path2, fname[:-4]+".npy")
        file_path_dict[dict_ptr] = (_img_path, _label_path)
        file_name_dict[dict_ptr] = f"{fname[:-len_image_extension]}"
        dict_ptr += 1
    elif split == "validation":
      _path = osp.join(self.source_domain_folder, "validation", "images")
      _path2 = osp.join(self.source_domain_folder, "validation", "labels")
      for fname in os.listdir(_path):
        _img_path = os.path.join(_path, fname)
        _label_path = os.path.join(_path2, fname[:-4]+".npy")
        file_path_dict[dict_ptr] = (_img_path, _label_path)
        file_name_dict[dict_ptr] = f"{fname[:-len_image_extension]}"
        dict_ptr += 1
    elif split == "test":
      _path = osp.join(self.source_domain_folder, "test", "images")
      for fname in os.listdir(_path):
        _img_path = os.path.join(_path, fname)
        file_path_dict[dict_ptr] = (_img_path)
        file_name_dict[dict_ptr] = f"{fname[:-len_image_extension]}"
        dict_ptr += 1
    else:
        raise Exception(f"split should only be `train`, `test`, or `validation`")


    self.source_domain_length = dict_ptr

    for i in range(len(self.folders)):
      _path = osp.join(self.target_domain_folder, self.folders[i], "synthetic")
      _path_2 = osp.join(self.target_domain_folder, self.folders[i], "mask-withskin")
      for image_name in os.listdir(_path):
        _img_path = osp.join(_path, image_name)
        _label_path = osp.join(_path_2, image_name)
        file_path_dict[dict_ptr] = (_img_path, _label_path)
        file_name_dict[dict_ptr] = f"{self.folders[i]}_{image_name[:-len_image_extension]}"
        dict_ptr += 1

    self.file_path_dict = file_path_dict
    self.file_name_dict = file_name_dict
    self.dict_ptr = dict_ptr

    #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS 
    #local Contrast limited adaptive histogram equalization algorithm
    self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

  def __len__(self):
    return self.dict_ptr

  def _compute_crop_size_for_openeds(self, height, width):
    return (height * self.E.config.h_over_w_ratio, width)

  def _perform_random_crop(self, pil_img: Image):
    """ Naive way of random crop

    Args:
        pil_img (Image): _description_

    Returns:
        Image: _description_
        Tuple[int]
    """
    w, h = pil_img.size
    starting_pixel = random.randint(80, 120) # Choose a starting pixel from 80, 120
    crop_info = (0, starting_pixel, w, starting_pixel + int(self.E.h_over_w_ratio * w)) # Pil doc: (left, top, right, bottom)
    crop_img = pil_img.crop(crop_info)
    return crop_img, crop_info

  def _perform_crop_with_crop_info(self, pil_img: Image, crop_info: Tuple[int]) -> Image:
    return pil_img.crop(crop_info)

  def _is_target_domain(self, idx):
    # Source domain class is 0
    # Target domain class is 1
    return idx >= self.source_domain_length

  def __getitem__(self, idx):
    # Dealing with corrupted data file. https://stackoverflow.com/a/71493112
    try:
      # Read image
      image_path = self.file_path_dict[idx][0]
      pil_img = Image.open(image_path).convert("L") # Convert to grayscale, shape (H, W)
      img = np.asarray(pil_img)
      
      # Read label
      if self.split != "test":
        label_path: str = self.file_path_dict[idx][1]
        if self._is_target_domain(idx):
          label_pil_img = Image.open(label_path)
          label_img: np.ndarray = np.asarray(label_pil_img) # (H, W, 3)
        else:
          label_img: np.ndarray = np.load(label_path)
    except:
      # Dealing with corrupted data file. https://stackoverflow.com/a/71493112
      idx = np.random.randint(0, len(self)-1)
      sample = self[idx]
      return sample

    [height, width] = img.shape[:2]

    if self.split != "test":
      
      assert label_img.shape[:2] == (height, width), "Unmatched label and image size"
      
      if self._is_target_domain(idx):
        label: np.ndarray = preprocess_label_sparse_np(label_img) # (H, W)
      else:
        label = label_img
      try:
        img, label = get_eye_horizontal_window(img, label, image_size=self.E.config.training_image_size[::-1])
      except:
        # Dealing with corrupted data file. https://stackoverflow.com/a/71493112
        idx = np.random.randint(0, len(self)-1)
        sample = self[idx]
        return sample
      tensor_label = MaskToTensor()(label) # Pytorch tensor (H, W)
    
    # After reading, process
    # PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS
    # Fixed gamma value
    table = 255.0*(np.linspace(0, 1, 256)**0.8)
    img_after = cv2.LUT(img, table)
    img = self.clahe.apply(np.array(np.uint8(img_after)))
    img = Image.fromarray(img)
    
    if self.split != 'test':
        
      ## This is for boundary aware cross entropy calculation
      spatial_weights = cv2.Canny(label.astype(np.uint8), 0, 3) / 255
      spatial_weights = cv2.dilate(spatial_weights, (3, 3), iterations = 1) * 20 # [0,20] - float - np
      
      # This is the implementation for the surface loss
      # Distance map for each class
      distance_map = [] 
      for i in range(0, 4):
        distance_map.append(one_hot2dist(label == i))
      distance_map = np.stack(distance_map, 0) #  [-1, 1], float, np
      
    if self.transform is not None:
      img = self.transform(img)
    
    if self.split == "test":
      return img, self.file_name_dict[idx], self._is_target_domain(idx)

    return img, tensor_label, spatial_weights, distance_map, self.file_name_dict[idx], torch.tensor(self._is_target_domain(idx)).unsqueeze(-1).float()


class DannDataset3(torch.utils.data.Dataset):
  """ This class different from the other danndataset class in that
  this class will take a list of source domain and a single target domain
  as parameters, and this also take h5 bigdata file.
  """
  def __init__(self, 
      experiment: Experiment, 
      split: str="train", 
      transform=None) -> None:

    self.E = experiment
    self.transform = transform
    # Source is synthetic eye imagery: list of synthetic dataset
    self.source_domains = self.E.config.source_domain
    # Target is real eye
    self.target_domain = self.E.config.target_domain
    self.split = split
    
    self.image_size = self.E.config.training_image_size ### (height, width)
    
    # "image", # image data
    # "resolution", # (width, height)
    # "path", # path to the image
    # "label",
    if split == "train":
      target_domain_data = h5py.File(self.target_domain, 'r')["train"]
      source_domains_data = [h5py.File(source_path, 'r')["train"] for source_path in self.source_domains]
    elif split == "validation":
      # target_domain_data = h5py.File(self.target_domain, 'r')["val"]
      # source_domains_data = [h5py.File(source_path, 'r')["val"] for source_path in self.source_domains]
      target_domain_data = h5py.File(self.target_domain, 'r')["train"]
      source_domains_data = [h5py.File(source_path, 'r')["train"] for source_path in self.source_domains]
    else:
      raise Exception(f"split should only be `train` or `validation`")

    self.idx_map = collections.defaultdict(int)
    ptr = 0

    # map target domain
    for i in range(len(target_domain_data["image"])):
      self.idx_map[ptr] = i
      ptr += 1
    
    # SAve target domain last pointer
    self.target_domain_length = ptr
    # (endpoint ptr of domain target, source1, source2, source3, ...)
    self.domain_endpoint = [ptr]

    # map all the source dmains
    for source_data in source_domains_data:
      for i in range(len(source_data["image"])):
        self.idx_map[ptr] = i
        ptr += 1
      self.domain_endpoint.append(ptr)
    self.len = ptr

    # free_memories(["target_domain_data", "source_domains_data"])
    del target_domain_data, source_domains_data

  def _read_data(self, idx):

    subset = None
    if self.split == "train":
      subset = "train"
    elif self.split == "validation":
      subset = "val"
    else:
      raise Exception(f"split should only be `train` or `validation`")

    domain_number = np.where(np.asarray(self.domain_endpoint) > idx)[0][0]
    
    # If is target domain
    if domain_number == 0:
      data_bank = h5py.File(self.target_domain, 'r')[subset]
      img = data_bank["image"][self.idx_map[idx]]
      label = data_bank["label"][self.idx_map[idx]]
      del data_bank
      # https://stackoverflow.com/a/1316793
      # gc.collect()
    else:
      # If it is source domain
      data_banks = [h5py.File(source_path, 'r')[subset] for source_path in self.source_domains]
      domain_number -= 1
      img = data_banks[domain_number]["image"][self.idx_map[idx]]
      label = data_banks[domain_number]["label"][self.idx_map[idx]]
      del data_banks
      # https://stackoverflow.com/a/1316793
      # gc.collect()
    
    # free_memories(["subset", "domain_number", "data_bank", "data_banks"])
    del subset, domain_number
    # https://stackoverflow.com/a/1316793
    # gc.collect()

    return img, label

  def __len__(self):
    return self.len

  def _is_target_domain(self, idx):
    # Source domain classes is 0
    # Target domain class is 1
    return idx < self.target_domain_length

  def __getitem__(self, idx):
    
    assert idx >= 0 and idx < self.__len__()

    img, label = self._read_data(idx)
    
    # pil_img = pil_img.resize(self.image_size[::-1]) # Since resizing in PIL take (width, height, instead of height, width)

    [height, width] = img.shape[:2]
    assert label.shape[:2] == (height, width), "Unmatched label and image size"
    label_tensor = MaskToTensor()(label) # Pytorch tensor (H, W)
    
    ## This is for boundary aware cross entropy calculation
    spatial_weights = cv2.Canny(label.astype(np.uint8), 0, 3) / 255
    spatial_weights = cv2.dilate(spatial_weights, (3, 3), iterations = 1) * 20
    
    # This is the implementation for the surface loss
    # Distance map for each class
    distance_map = []
    for i in range(0, 4):
      distance_map.append(one_hot2dist(label == i))
    distance_map = np.stack(distance_map, 0)
      
    img = self.transform(img)

    # Clean memory
    del label

    return img, label_tensor, spatial_weights, distance_map, torch.tensor(self._is_target_domain(idx)).unsqueeze(-1).float()

  def __str__(self) -> str:
    return f"[DannDataset3] target_domain_length: {self.target_domain_length}, ptr_endpoints (target, source1, source2, ...): {self.domain_endpoint}"


class CVDataset(torch.utils.data.Dataset):
  """ This class different from the other danndataset class in that
  this class will take a list of source domain and a single target domain
  as parameters, and this also take h5 bigdata file. This class only take one data file
  """
  def __init__(self,
      data_path: str,
      split: str="train", 
      transform=None,
      data_point_limit:int =None,
      shuffle=False,
      n_segments: int=None,
      include_idx: List[int]=None) -> None:

    self.transform = transform
    self.data_path = data_path
    self.split = split

    self.shuffle = shuffle 

    if split == "train":
      self.data = h5py.File(self.data_path, 'r')["train"]
    elif split == "validation": # The dataset now divide train and validation on fold, not this keyword. This keyword is rather specifying how to process the data
      # self.data = h5py.File(self.data_path, 'r')["val"]
      self.data = h5py.File(self.data_path, 'r')["train"] # The dataset now divide train and validation on fold, not this keyword. This keyword is rather specifying how to process the data
    else:
      raise Exception(f"split should only be `train` or `validation`")

    self.len = self.data["image"].__len__()

    # Maximum length of the data bank
    self.max_len = self.len

    # Data accessor
    self.mini_dataset_indices = [i for i in range(self.max_len)][:self.len]
    if self.shuffle:
      self.mini_dataset_indices = random.sample([i for i in range(self.max_len)], self.len)

    # K-fold
    if n_segments is not None and include_idx is not None:
      tmp_idx = []
      len_each_segment = self.len // n_segments
      for id in include_idx:
        assert 0 <= id < n_segments
        min_id = id * len_each_segment
        max_id = min((id + 1) * len_each_segment, self.len)
        tmp_idx = tmp_idx + self.mini_dataset_indices[min_id:max_id]
      self.mini_dataset_indices = tmp_idx
      self.len = len(self.mini_dataset_indices)
    
    # We want to limit the number of datapoints after process the segmenta data
    if data_point_limit is not None:
      self.len = min(data_point_limit, self.len)

    #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS 
    #local Contrast limited adaptive histogram equalization algorithm
    self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

  def __len__(self):
    return self.len

  def get_actual_databank_idx(self, idx):
    return self.mini_dataset_indices[idx]

  def getitem_with_actual_databank_idx(self, databank_idx):
    # print(idx)
    assert databank_idx >= 0 and databank_idx < self.max_len
    img = self.data["image"][databank_idx]
    label = self.data["label"][databank_idx]
    
    [height, width] = img.shape[:2]
    assert label.shape[:2] == (height, width), "Unmatched label and image size"
    
    ## This is for boundary aware cross entropy calculation
    spatial_weights = cv2.Canny(label.astype(np.uint8), 0, 3) / 255
    spatial_weights = cv2.dilate(spatial_weights, (3, 3), iterations = 1) * 20
    spatial_weights = torch.tensor(np.asarray(spatial_weights))
    
    # This is the implementation for the surface loss
    # Distance map for each class
    distance_map = []
    for i in range(0, 4):
      distance_map.append(one_hot2dist(label == i))
    distance_map = np.stack(distance_map, 0)
    distance_map = torch.tensor(np.float32(distance_map))

    # print(f"img shape: {img.shape}")
    if self.transform is not None:
      if self.split == 'train':
        if random.random() < 0.2:
        # if True: 
          img = Starburst_augment()(np.array(img))  
        if random.random() < 0.2: 
          img = Line_augment()(np.array(img))    
        if random.random() < 0.2:
          img = Gaussian_blur()(np.array(img))   
        if random.random() < 0.2:
          img, label = Translation()(np.array(img),np.array(label))

    img = self.clahe.apply(np.array(np.uint8(img)))    
    img = Image.fromarray(img)
        
    img = self.transform(img)
    label_tensor = MaskToTensor()(label) # Pytorch tensor (H, W)

    # Clean memory
    # free_memories(["label", "height", "width"])
    del label, height, width
    # https://stackoverflow.com/a/1316793
    # gc.collect()

    # print(f"img: {img.shape}, ")

    return img, label_tensor, spatial_weights, distance_map

  def __getitem__(self, idx):
    assert idx >= 0 and idx < self.len
    return self.getitem_with_actual_databank_idx(self.get_actual_databank_idx(idx))

class DannDataset3Short(DannDataset3):
  def __init__(self, experiment: Experiment, split: str = "train", transform=None, data_point_limit: int=20000) -> None:
    super().__init__(experiment, split, transform)
    self.len = min(data_point_limit, self.len)


class CVDatasetWithIndex(CVDataset):
  """ WITH DATABNK INDEX, NOT INTERNAL INDEX BECAUSE 
    THE INDEX CAN BE SHUFFLED MANY TIMES
  """
  def __getitem__(self, idx):
    return (*super().__getitem__(idx), self.get_actual_databank_idx(idx))

class IncrementDataset(Dataset):
  def __init__(self, dataset: CVDataset) -> None:
    self.dataset = dataset
    # TODO: Should we create and deque with maxlen here?
    self.indices = []

  def __len__(self):
    return len(self.indices)

  def add_data(self, databank_idx):
    if isinstance(databank_idx, int):
      self.indices.append(databank_idx)
    elif isinstance(databank_idx, list):
      self.indices = self.indices + databank_idx
    else:
      raise Exception("idx shoule only be int or list of integer")
  
  def reset(self):
    self.indices = []

  def __getitem__(self, idx):
    """ Return the parent's get data for the saved index in the list of indices added

    Args:
      idx (int):

    Returns:
      img (torch.Tensor): 
      label (torch.Tensor):
    """
    # print(f"idx: {idx}, actual idx: {self.indices[idx]}") # Tested!
    return self.dataset.getitem_with_actual_databank_idx(self.indices[idx])


class PairedImageDataset(torch.utils.data.Dataset):
  """ This class different from the other danndataset class in that
  this class will take a list of source domain and a single target domain
  as parameters, and this also take h5 bigdata file. This class only take one data file
  """
  def __init__(self,
      E: Experiment,
      data_path_A: str,
      data_path_B: str,
      split: str="train",
      transform=None,
      data_point_limit:int =None) -> None:
    self.E = E
    self.transform = transform
    self.data_path_A = data_path_A
    self.data_path_B = data_path_B
    self.split = split
    self.data_point_limit = data_point_limit
    
    # "image", # image data
    # "resolution", # (width, height)
    # "path", # path to the image
    # "label",
    if split == "train":
      self.data_A = h5py.File(self.data_path_A, 'r')["train"]
      self.data_B = h5py.File(self.data_path_B, 'r')["train"]
    elif split == "validation":
      self.data_A = h5py.File(self.data_path_A, 'r')["train"] # val
      self.data_B = h5py.File(self.data_path_B, 'r')["train"] # val
    else:
      raise Exception(f"split should only be `train` or `validation`")

  def __len__(self):
    return min(self.data_A["image"].__len__(), self.data_B["image"].__len__())

  def __getitem__(self, idx):
    # Unaligned data pair

    img_a = self.data_A["image"][idx % self.data_A["image"].__len__()]
    img_b = self.data_B["image"][np.random.randint(0, len(self.data_B["image"])-1)]
    
    [height, width] = img_a.shape[:2]
    img_a = Image.fromarray(img_a)
    img_b = Image.fromarray(img_b)
    
    img_a = self.transform(img_a)
    img_b = self.transform(img_b)

    return {'A': img_a, 'B': img_b}

class PairedImageDatasetWithLabel(PairedImageDataset):
  def __getitem__(self, idx):
    id_a = idx % self.data_A["image"].__len__()
    id_b = np.random.randint(0, len(self.data_B["image"])-1)
    
    # load images
    img_a = self.data_A["image"][id_a]
    img_b = self.data_B["image"][id_b]
    [height, width] = img_a.shape[:2]
    img_a = Image.fromarray(img_a)
    img_b = Image.fromarray(img_b)
    img_a = self.transform(img_a)
    img_b = self.transform(img_b)

    label_a = self.data_A["label"][id_a]
    label_b = self.data_B["label"][id_b]

    assert label_a.shape[:2] == (height, width), "Unmatched label and image size"
    assert label_b.shape[:2] == (height, width), "Unmatched label and image size"
    label_tensor_a = MaskToTensor()(label_a) # Pytorch tensor (H, W)
    label_tensor_b = MaskToTensor()(label_b) # Pytorch tensor (H, W)

    return {'A': (img_a, label_tensor_a), 'B': (img_b, label_tensor_b)}

class PairedImageDatasetWithLabelAndSpatialAndDistance(PairedImageDataset):
  def __init__(self, E: Experiment, data_path_A: str, data_path_B: str, split: str = "train", transform=None, data_point_limit: int = None, n_classes=5) -> None:
    super().__init__(E, data_path_A, data_path_B, split, transform, data_point_limit)
    self.n_classes = n_classes

  def __getitem__(self, idx):
    id_a = idx % self.data_A["image"].__len__()
    id_b = np.random.randint(0, len(self.data_B["image"])-1)
    
    # load images
    img_a = self.data_A["image"][id_a]
    img_b = self.data_B["image"][id_b]
    [height, width] = img_a.shape[:2]
    img_a = Image.fromarray(img_a)
    img_b = Image.fromarray(img_b)
    img_a = self.transform(img_a)
    img_b = self.transform(img_b)

    label_a = self.data_A["label"][id_a]
    label_b = self.data_B["label"][id_b]

    assert label_a.shape[:2] == (height, width), "Unmatched label and image size"
    assert label_b.shape[:2] == (height, width), "Unmatched label and image size"
    label_tensor_a = MaskToTensor()(label_a) # Pytorch tensor (H, W)
    label_tensor_b = MaskToTensor()(label_b) # Pytorch tensor (H, W)

    ## This is for boundary aware cross entropy calculation
    spatial_weights_a = cv2.Canny(label_a.astype(np.uint8), 0, 3) / 255
    spatial_weights_a = cv2.dilate(spatial_weights_a, (3, 3), iterations = 1) * 20
    
    # This is the implementation for the surface loss
    # Distance map for each class
    distance_map_a = []
    for i in range(0, self.n_classes):
      distance_map_a.append(one_hot2dist(label_a == i))
    distance_map_a = np.stack(distance_map_a, 0)

    ## This is for boundary aware cross entropy calculation
    spatial_weights_b = cv2.Canny(label_b.astype(np.uint8), 0, 3) / 255
    spatial_weights_b = cv2.dilate(spatial_weights_b, (3, 3), iterations = 1) * 20
    
    # This is the implementation for the surface loss
    # Distance map for each class
    distance_map_b = []
    for i in range(0, self.n_classes):
      distance_map_b.append(one_hot2dist(label_b == i))
    distance_map_b = np.stack(distance_map_b, 0)


    return {'A': (img_a, label_tensor_a, spatial_weights_a, distance_map_a), 'B': (img_b, label_tensor_b, spatial_weights_b, distance_map_b)}


def visualize_dataset(path, split="train", n=16, transform=transform_sequence):
  data = h5py.File(path)
  length = data[split]["image"].__len__()
  idx = np.random.choice(length, size=(n,))
  idx.sort()
  sample_data = data[split]["image"][idx]
  transformed_data = []
  for img in sample_data:
    trans_img = transform(img)
    transformed_data.append(trans_img)
  transformed_data = torch.stack(transformed_data, dim=0)
  # print(transformed_data.shape)
  sample_img = torchvision.utils.make_grid(transformed_data, normalize=True, nrow=n//4)
  fig = plt.figure()
  plt.imshow(sample_img.cpu().permute(1,2,0))
  plt.axis('off')
  plt.show()
  plt.close(fig)