from abc import ABC, abstractmethod
import random
from typing import List
import numpy as np
import cv2
import os
import deepdish
from PIL import Image
import h5py

from ..utils import preprocess_label_sparse_np, preprocess_chengyi_label_sparse_np
from ..utils.image import get_eye_horizontal_window, show_img

class Extractor(ABC):
  def __init__(self, dataset_dir: str, target_path: str) -> None:
    super().__init__()
    self.dataset_dir = dataset_dir
    self.target_path = target_path

  @abstractmethod
  def extract(self):
    pass

  @staticmethod
  def initialize_data():
    data = {
      "train": {k: [] for k in [
        "image", # image data
        "resolution", # (width, height)
        "path", # path to the image
        "label", # label with skin
      ]},
      "val": {k: [] for k in [
        "image", # image data
        "resolution", # (width, height)
        "path", # path to the imge
        "label", # label with skin
      ]},
    }

    return data

class OpenEDSExtractor(Extractor):
  """ ##### CLASS ORDER ###########
      ##### Backgrund: 0
      ##### Sclera: 1
      ##### Iris: 2
      ##### PUPIL: 3

  Args:
      Extractor (_type_): _description_
  """
  def __init__(self, dataset_dir: str, target_path: str, image_size=(320, 240), val_with_train: bool=False, keep_original=False) -> None:
    """ Extract and put everything in train,val,test set, resize and preprocess features

    Args:
      dataset_dir (str): _description_
      target_dir (str): _description_
      image_size (tuple, optional): (width, height). Defaults to (640, 480).
      val_with_train (bool): specify that training data and val data is put together 
        in the train set of the h5, the val data is still specifically put in the val set
    """
    super().__init__(dataset_dir, target_path)

    self.train_imgs = os.path.join(dataset_dir, "train", "images")
    self.val_imgs = os.path.join(dataset_dir, "validation", "images")
    self.test_imgs = os.path.join(dataset_dir, "test", "images")
    self.train_labels = os.path.join(dataset_dir, "train", "labels")
    self.val_labels = os.path.join(dataset_dir, "validation", "labels")
    self.image_size = image_size
    self.val_with_train = val_with_train
    self.keep_original = keep_original

    # self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

    self.data = Extractor.initialize_data()

  def extract(self, resizing=True):

    # Extract training dir
    trn = os.listdir(self.train_imgs)
    n = len(trn)
    for i, fname in enumerate(trn):
      if i % 600 == 0:
        print(f"Extracting train: {i+1}/{n}")
      try:
        img_name, _ = os.path.splitext(fname)
        img_path = os.path.join(self.train_imgs, fname)
        label_path = os.path.join(self.train_labels, img_name + ".npy")
        img = cv2.imread(img_path, 0) # uint8 0-255
        label = np.load(label_path) # 0-1-2-3
        # print(label.shape) # 640, 400
        # return
      except:
        # Corrupted data file
        continue
      
      if not self.keep_original:
        if resizing:
          img, label = get_eye_horizontal_window(img, label, image_size=self.image_size) # uint8 0-255
        else:
          img, label = get_eye_horizontal_window(img, label, image_size=None) # uint8 0-255
      else:
        if resizing:
          img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LANCZOS4)
          label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)

      # Ratio w:h == 5:4

      [height, width] = img.shape[:2]
      assert label.shape[:2] == (height, width)
      

      if (width, height) != self.image_size:
        print("An instance does not match desired image size, skipping")
        print(f"width: {width}, height: {height}, self img size: {self.image_size}")
        show_img(img)
        continue

      # show_img()
      

      # Fixed gamma value
      # table = 255.0*(np.linspace(0, 1, 256)**0.8)
      # clahe = cv2.LUT(img, table)
      # img = self.clahe.apply(np.array(np.uint8(clahe)))
      # img = Image.fromarray(img)
      # print(img.size)

      ## Save data ##
      self.data["train"]["image"].append(img)
      self.data["train"]["resolution"].append((width, height))
      self.data["train"]["path"].append(img_path)
      self.data["train"]["label"].append(label)

    trn = os.listdir(self.val_imgs)
    n = len(trn)
    # Extract val dir
    for i, fname in enumerate(trn):
      if i % 600 == 0:
        print(f"Extracting val: {i+1}/{n}")
      try:
        img_name, _ = os.path.splitext(fname)
        img_path = os.path.join(self.val_imgs, fname)
        label_path = os.path.join(self.val_labels, img_name + ".npy")
        img = cv2.imread(img_path, 0) # uint8 0-255
        label = np.load(label_path) # 0-1-2-3
      except:
        # Corrupted data file
        continue

      if not self.keep_original:
        if resizing:
          img, label = get_eye_horizontal_window(img, label, image_size=self.image_size) # uint8 0-255
        else:
          img, label = get_eye_horizontal_window(img, label, image_size=None) # uint8 0-255
      else:
        if resizing:
          img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LANCZOS4)
          label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)
      
      [height, width] = img.shape[:2]
      assert label.shape[:2] == (height, width)

      if (width, height) != self.image_size:
        print("An instance does not match desired image size, skipping")
        continue

      # Fixed gamma value
      # table = 255.0*(np.linspace(0, 1, 256)**0.8)
      # clahe = cv2.LUT(img, table)
      # img = self.clahe.apply(np.array(np.uint8(clahe)))
      # img = Image.fromarray(img)
      # print(img.size)

      ## Save data ##
      self.data["val"]["image"].append(img)
      self.data["val"]["resolution"].append((width, height))
      self.data["val"]["path"].append(img_path)
      self.data["val"]["label"].append(label)

      if self.val_with_train:
        self.data["train"]["image"].append(img)
        self.data["train"]["resolution"].append((width, height))
        self.data["train"]["path"].append(img_path)
        self.data["train"]["label"].append(label)
    
    ### After looping done, stack all data using np and save
    self.data["train"]["image"] = np.stack(self.data["train"]["image"], axis=0)
    self.data["train"]["resolution"] = np.stack(self.data["train"]["resolution"], axis=0)
    self.data["train"]["path"] = np.stack(self.data["train"]["path"], axis=0)
    self.data["train"]["label"] = np.stack(self.data["train"]["label"], axis=0)
    
    self.data["val"]["image"] = np.stack(self.data["val"]["image"], axis=0)
    self.data["val"]["resolution"] = np.stack(self.data["val"]["resolution"], axis=0)
    self.data["val"]["path"] = np.stack(self.data["val"]["path"], axis=0)
    self.data["val"]["label"] = np.stack(self.data["val"]["label"], axis=0)
    
    deepdish.io.save(os.path.join(self.target_path), self.data)


class RITEyesExtractor(Extractor):
  ##### CLASS ORDER ###########
  ##### Backgrund: 0
  ##### Sclera: 1
  ##### Iris: 2
  ##### PUPIL: 3
  def __init__(self, dataset_dir: str, target_path: str, image_size=(320, 240), val_with_train=False) -> None:
    """ Extract and put everything in train,val,test set, resize and preprocess features

    Args:
      dataset_dir (str): _description_
      target_dir (str): _description_
      image_size (tuple, optional): (width, height). Defaults to (640, 480).
      val_with_train (bool): specify that training data and val data is put together 
        in the train set of the h5, the val data is still specifically put in the val set
    """
    super().__init__(dataset_dir, target_path)

    self.dataset_dir = dataset_dir
    self.train_folder = [i for i in range(1, 20, 1)]
    self.val_folder = [i for i in range(20, 25, 1)]
    self.image_size = image_size
    self.val_with_train = val_with_train

    # self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

    self.data = Extractor.initialize_data()

  def extract(self, resizing=True):

    dict_ptr = 0
    for i in range(len(self.train_folder)):
      
      # _path = os.path.join(self.dataset_dir, f"{self.train_folder[i]}", "synthetic")
      # _path_2 = os.path.join(self.dataset_dir, f"{self.train_folder[i]}", "mask-withskin")
      _path = os.path.join(self.dataset_dir, "synthetic", f"{self.train_folder[i]}")
      _path_2 = os.path.join(self.dataset_dir, "Mask_withskin", f"{self.train_folder[i]}")
      for image_name in os.listdir(_path):
        if dict_ptr % 600 == 0:
          print(f"Extracting train: {dict_ptr}")
        _img_path = os.path.join(_path, image_name)
        _label_path = os.path.join(_path_2, image_name)
        # try:
        img = cv2.imread(_img_path, 0) # uint8 0-255
        label = Image.open(_label_path)
        label: np.ndarray = np.asarray(label) # (480, 640) uint8, [0,255]
        # except:
        #   # Corrupted data file
        #   continue
        
        if resizing:
          img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LANCZOS4)
          label = cv2.resize(label, self.image_size)

        img = np.asarray(img) # (240, 320) [0, 255] uint8
        label = np.asarray(label) # (240, 320) 0-1-2-3
        # label = preprocess_label_sparse_np(label) # (480, 640)
        label = preprocess_chengyi_label_sparse_np(label)

        [height, width] = img.shape[:2]

        if (width, height) != self.image_size:
          print("An instance does not match desired image size, skipping")
          continue

        ### Append data
        self.data["train"]["image"].append(img)
        self.data["train"]["resolution"].append((width, height))
        self.data["train"]["path"].append(_img_path)
        self.data["train"]["label"].append(label)

        dict_ptr += 1

    dict_ptr = 0
    for i in range(len(self.val_folder)):
      # _path = os.path.join(self.dataset_dir, f"{self.val_folder[i]}", "synthetic")
      # _path_2 = os.path.join(self.dataset_dir, f"{self.val_folder[i]}", "mask-withskin")
      _path = os.path.join(self.dataset_dir, "synthetic", f"{self.val_folder[i]}")
      _path_2 = os.path.join(self.dataset_dir, "Mask_withskin", f"{self.val_folder[i]}")
      for image_name in os.listdir(_path):
        if dict_ptr % 600 == 0:
          print(f"Extracting val: {dict_ptr}")
        _img_path = os.path.join(_path, image_name)
        _label_path = os.path.join(_path_2, image_name)
        try:
          img = cv2.imread(_img_path, 0) # uint8 0-255
          label = Image.open(_label_path)
          label: np.ndarray = np.asarray(label) # (480, 640) uint8, [0,255]
        except:
          # Corrupted data file
          continue
        
        if resizing:
          img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LANCZOS4)
          label = cv2.resize(label, self.image_size)

        img = np.asarray(img) # (240, 320) [0, 255] uint8
        label = np.asarray(label) # (240, 320) 0-1-2-3
        # label = preprocess_label_sparse_np(label) # (480, 640)
        label = preprocess_chengyi_label_sparse_np(label)

        [height, width] = img.shape[:2]

        if (width, height) != self.image_size:
          print("An instance does not match desired image size, skipping")
          continue

        ### Append data
        self.data["val"]["image"].append(img)
        self.data["val"]["resolution"].append((width, height))
        self.data["val"]["path"].append(_img_path)
        self.data["val"]["label"].append(label)

        if self.val_with_train:
          self.data["train"]["image"].append(img)
          self.data["train"]["resolution"].append((width, height))
          self.data["train"]["path"].append(_img_path)
          self.data["train"]["label"].append(label)

        dict_ptr += 1

    ### After looping done, stack all data using np and save
    self.data["train"]["image"] = np.stack(self.data["train"]["image"], axis=0)
    self.data["train"]["resolution"] = np.stack(self.data["train"]["resolution"], axis=0)
    self.data["train"]["path"] = np.stack(self.data["train"]["path"], axis=0)
    self.data["train"]["label"] = np.stack(self.data["train"]["label"], axis=0)
    
    self.data["val"]["image"] = np.stack(self.data["val"]["image"], axis=0)
    self.data["val"]["resolution"] = np.stack(self.data["val"]["resolution"], axis=0)
    self.data["val"]["path"] = np.stack(self.data["val"]["path"], axis=0)
    self.data["val"]["label"] = np.stack(self.data["val"]["label"], axis=0)
    
    deepdish.io.save(self.target_path, self.data)

class CustomFolderRITEyesExtractor(RITEyesExtractor):
  ##### CLASS ORDER ###########
  ##### Backgrund: 0
  ##### Sclera: 1
  ##### Iris: 2
  ##### PUPIL: 3
  """ Configure the train and val folder of this class that match the desired train and val folder

  Args:
      RITEyesExtractor (_type_): _description_
  """
  def __init__(self, train_folder: List[int], val_folder: List[int], dataset_dir: str, target_path: str, image_size=(320, 240), val_with_train=False,) -> None:
    super().__init__(dataset_dir, target_path, image_size, val_with_train)

    self.train_folder = train_folder # [0] # folder name 0 in the dataset folder
    self.val_folder = val_folder # [1] # fodler name 1

class Fusioner:
  def __init__(self, h5_list: List[str], train_length=None, val_length=None) -> None:
    self.h5_list = h5_list
    if train_length is None:
      self.train_length = []
    else:
      self.train_length = train_length

    if val_length is None:
      self.val_length = []
    else:
      self.val_length = val_length
    self.data = Extractor.initialize_data()

  def fuse(self):

    train_indices = []
    val_indices = []
    total_train_length = 0
    total_val_length = 0
    print("Shuffling...")
    for i, fname in enumerate(self.h5_list):
      data = h5py.File(fname, 'r')
      length_train = len(data["train"]["image"])
      trainn = [i for i in range(length_train)]
      random.shuffle(trainn)
      if i >= len(self.train_length):
        self.train_length.append(length_train)
      trainn = trainn[:min(len(trainn), self.train_length[i])]
      total_train_length += len(trainn)
      train_indices.append(trainn)

      length_val = len(data["val"]["image"])
      vall = [i for i in range(length_val)]
      random.shuffle(vall)
      if i >= len(self.val_length):
        self.val_length.append(length_val)
      vall = vall[:min(len(vall), self.val_length[i])]
      total_val_length += len(vall)
      val_indices.append(vall)

    print(f"There are a total of {total_train_length} training data points, and {total_val_length} val data points.")

    print("Fusioning train indices...")
    # Train indices
    i = 0
    while len(train_indices) > 0:
      which_data_file = random.randint(0, len(train_indices) - 1)

      # If list is empty, pop the empty list, continue
      if len(train_indices[which_data_file]) == 0:
        train_indices.pop(which_data_file)
        continue
      else:
        idx = train_indices[which_data_file].pop(0)
        data = h5py.File(self.h5_list[which_data_file], 'r')["train"]
        # Append data to global data
        self.data["train"]["image"].append(data["image"][idx])
        self.data["train"]["resolution"].append(data["resolution"][idx])
        self.data["train"]["path"].append(data["path"][idx])
        self.data["train"]["label"].append(data["label"][idx])

      if i % 200 == 0:
        print(f"[Fusing] Train: {i}/{total_train_length}")

      i += 1

    print("Fusioning val indices...")
    # Val indices
    i = 0
    while len(val_indices) > 0:
      which_data_file = random.randint(0, len(val_indices) - 1)

      # If list is empty, pop the empty list, continue
      if len(val_indices[which_data_file]) == 0:
        val_indices.pop(which_data_file)
        continue
      else:
        idx = val_indices[which_data_file].pop(0)
        data = h5py.File(self.h5_list[which_data_file], 'r')["val"]
        # Append data to global data
        self.data["val"]["image"].append(data["image"][idx])
        self.data["val"]["resolution"].append(data["resolution"][idx])
        self.data["val"]["path"].append(data["path"][idx])
        self.data["val"]["label"].append(data["label"][idx])

      if i % 200 == 0:
        print(f"[Fusing] Val: {i}/{total_val_length}")
      i += 1

  
  def export_data(self, target_path):
    print("Exporting data...")
    self.data["train"]["image"] = np.stack(self.data["train"]["image"], axis=0)
    self.data["train"]["resolution"] = np.stack(self.data["train"]["resolution"], axis=0)
    self.data["train"]["path"] = np.stack(self.data["train"]["path"], axis=0)
    self.data["train"]["label"] = np.stack(self.data["train"]["label"], axis=0)
    
    self.data["val"]["image"] = np.stack(self.data["val"]["image"], axis=0)
    self.data["val"]["resolution"] = np.stack(self.data["val"]["resolution"], axis=0)
    self.data["val"]["path"] = np.stack(self.data["val"]["path"], axis=0)
    self.data["val"]["label"] = np.stack(self.data["val"]["label"], axis=0)
    
    deepdish.io.save(os.path.join(target_path), self.data)


class RITEyesNewPipelineExtractor(Extractor):
  ##### CLASS ORDER ###########
  ##### Backgrund: 0
  ##### Sclera: 1
  ##### Iris: 2
  ##### PUPIL: 3
  def __init__(self, dataset_dir: str, target_path: str, image_size=(320, 240), val_with_train=False) -> None:
    """ Extract and put everything in train,val,test set, resize and preprocess features

    Args:
      dataset_dir (str): _description_
      target_dir (str): _description_
      image_size (tuple, optional): (width, height). Defaults to (640, 480).
      val_with_train (bool): specify that training data and val data is put together 
        in the train set of the h5, the val data is still specifically put in the val set
    """
    super().__init__(dataset_dir, target_path)

    self.dataset_dir = dataset_dir
    self.train_folder = ["head1", "head2", "head3", "head4", "head5"]
    self.val_folder = ["head5"]
    self.image_size = image_size
    self.val_with_train = val_with_train

    # self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

    self.data = Extractor.initialize_data()

  def extract(self, resizing=True):

    dict_ptr = 0
    for i in range(len(self.train_folder)):
      for eye_id in range(2): # eye 0 is the right eye (which is flipped by mistake in the pipeline), eye 1 is the left eye
        # _path = os.path.join(self.dataset_dir, f"{self.train_folder[i]}", "synthetic")
        # _path_2 = os.path.join(self.dataset_dir, f"{self.train_folder[i]}", "mask-withskin")
        _path = os.path.join(self.dataset_dir, f"{self.train_folder[i]}", "synthetic", f"{eye_id}")
        _path_2 = os.path.join(self.dataset_dir, f"{self.train_folder[i]}", "Mask_withskin", f"{eye_id}")
        for image_name in os.listdir(_path):
          if dict_ptr % 600 == 0:
            print(f"Extracting train: {dict_ptr}")
          _img_path = os.path.join(_path, image_name)
          _label_path = os.path.join(_path_2, image_name)
          # try:
          img = cv2.imread(_img_path, 0) # uint8 0-255
          label = Image.open(_label_path)

          if eye_id == 0: # if right eye, then flip both image and label
            img = cv2.rotate(img, cv2.ROTATE_180)
            label = label.rotate(180)

          label: np.ndarray = np.asarray(label) # (480, 640) uint8, [0,255]
          # except:
          #   # Corrupted data file
          #   continue
          
          if resizing:
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LANCZOS4)
            label = cv2.resize(label, self.image_size)

          img = np.asarray(img) # (240, 320) [0, 255] uint8
          label = np.asarray(label) # (240, 320) 0-1-2-3
          # label = preprocess_label_sparse_np(label) # (480, 640)
          label = preprocess_chengyi_label_sparse_np(label)

          [height, width] = img.shape[:2]

          if (width, height) != self.image_size:
            print("An instance does not match desired image size, skipping")
            continue

          ### Append data
          self.data["train"]["image"].append(img)
          self.data["train"]["resolution"].append((width, height))
          self.data["train"]["path"].append(_img_path)
          self.data["train"]["label"].append(label)

          dict_ptr += 1

    dict_ptr = 0
    for i in range(len(self.val_folder)):
      for eye_id in range(2): # eye 0 is the right eye (which is flipped by mistake in the pipeline), eye 1 is the left eye
        # _path = os.path.join(self.dataset_dir, f"{self.val_folder[i]}", "synthetic")
        # _path_2 = os.path.join(self.dataset_dir, f"{self.val_folder[i]}", "mask-withskin")
        _path = os.path.join(self.dataset_dir, f"{self.val_folder[i]}", "synthetic", f"{eye_id}")
        _path_2 = os.path.join(self.dataset_dir, f"{self.val_folder[i]}", "Mask_withskin", f"{eye_id}")
        for image_name in os.listdir(_path):
          if dict_ptr % 600 == 0:
            print(f"Extracting val: {dict_ptr}")
          _img_path = os.path.join(_path, image_name)
          _label_path = os.path.join(_path_2, image_name)
          try:
            img = cv2.imread(_img_path, 0) # uint8 0-255
            label: Image = Image.open(_label_path)
            
            if eye_id == 0: # if right eye, then flip both image and label
              img = cv2.rotate(img, cv2.ROTATE_180)
              label = label.rotate(180)

            label: np.ndarray = np.asarray(label) # (480, 640) uint8, [0,255]

          except:
            # Corrupted data file
            continue
          
          if resizing:
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LANCZOS4)
            label = cv2.resize(label, self.image_size)

          img = np.asarray(img) # (240, 320) [0, 255] uint8
          label = np.asarray(label) # (240, 320) 0-1-2-3
          # label = preprocess_label_sparse_np(label) # (480, 640)
          label = preprocess_chengyi_label_sparse_np(label)

          [height, width] = img.shape[:2]

          if (width, height) != self.image_size:
            print("An instance does not match desired image size, skipping")
            continue

          ### Append data
          self.data["val"]["image"].append(img)
          self.data["val"]["resolution"].append((width, height))
          self.data["val"]["path"].append(_img_path)
          self.data["val"]["label"].append(label)

          if self.val_with_train:
            self.data["train"]["image"].append(img)
            self.data["train"]["resolution"].append((width, height))
            self.data["train"]["path"].append(_img_path)
            self.data["train"]["label"].append(label)

          dict_ptr += 1

    ### After looping done, stack all data using np and save
    self.data["train"]["image"] = np.stack(self.data["train"]["image"], axis=0)
    self.data["train"]["resolution"] = np.stack(self.data["train"]["resolution"], axis=0)
    self.data["train"]["path"] = np.stack(self.data["train"]["path"], axis=0)
    self.data["train"]["label"] = np.stack(self.data["train"]["label"], axis=0)
    
    self.data["val"]["image"] = np.stack(self.data["val"]["image"], axis=0)
    self.data["val"]["resolution"] = np.stack(self.data["val"]["resolution"], axis=0)
    self.data["val"]["path"] = np.stack(self.data["val"]["path"], axis=0)
    self.data["val"]["label"] = np.stack(self.data["val"]["label"], axis=0)
    
    deepdish.io.save(self.target_path, self.data)

class OptimizedRITEyesNewPipelineExtractor(RITEyesNewPipelineExtractor):
  def extract(self, resizing=False):

    
    chunk_size = 500
    img_shape = (self.image_size[1], self.image_size[0])

    with h5py.File(self.target_path, 'w') as f:

      # For training dataset
      train = f.create_group("train")
      image = train.create_dataset('image', shape=(chunk_size, *img_shape), maxshape=(None, *img_shape),
                                chunks=(chunk_size, *img_shape), dtype=np.uint8)
      resolution = train.create_dataset('resolution', shape=(chunk_size, 2), maxshape=(None, 2),
                                chunks=(chunk_size, 2), dtype=np.uint8)
      # path = train.create_dataset('path', shape=(chunk_size), maxshape=(None,),
      #                           chunks=(chunk_size))
      label_dataset = train.create_dataset('label', shape=(chunk_size, *img_shape), maxshape=(None, *img_shape),
                                chunks=(chunk_size, *img_shape), dtype=np.uint8)
      writing_data = Extractor.initialize_data()
      row_count = 0
      i = 0
      count = 0


      dict_ptr = 0
      for folder_id in range(len(self.train_folder)):
        for eye_id in range(2): # eye 0 is the right eye (which is flipped by mistake in the pipeline), eye 1 is the left eye
          _path = os.path.join(self.dataset_dir, f"{self.train_folder[folder_id]}", "synthetic", f"{eye_id}")
          _path_2 = os.path.join(self.dataset_dir, f"{self.train_folder[folder_id]}", "Mask_withskin", f"{eye_id}")
          for image_name in os.listdir(_path):
            
            # Write data to h5 file every batch of 500
            if count % chunk_size == 0:
              print(f"batch {i}")
              if count != 0:
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
                label_dataset.resize(row_count + writing_data["train"]["label"].shape[0], axis=0)
                label_dataset[row_count:] = writing_data["train"]["label"]
                row_count += writing_data["train"]["image"].shape[0]
                writing_data = Extractor.initialize_data()

            #### Process the Images ####
            if dict_ptr % 500 == 0:
              print(f"Extracting train: {dict_ptr}")
            _img_path = os.path.join(_path, image_name)
            _label_path = os.path.join(_path_2, image_name)
            # try:
            img = cv2.imread(_img_path, 0) # uint8 0-255
            label = Image.open(_label_path)

            if eye_id == 0: # if right eye, then flip both image and label
              img = cv2.rotate(img, cv2.ROTATE_180)
              label = label.rotate(180)

            label: np.ndarray = np.asarray(label) # (480, 640) uint8, [0,255]
            # except:
            #   # Corrupted data file
            #   continue
            
            if resizing:
              img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LANCZOS4)
              label = cv2.resize(label, self.image_size)

            img = np.asarray(img) # (240, 320) [0, 255] uint8
            label = np.asarray(label) # (240, 320) 0-1-2-3
            # label = preprocess_label_sparse_np(label) # (480, 640)
            label = preprocess_chengyi_label_sparse_np(label)

            [height, width] = img.shape[:2]

            if (width, height) != self.image_size:
              print("An instance does not match desired image size, skipping")
              continue

            ### Append data
            writing_data["train"]["image"].append(img)
            writing_data["train"]["resolution"].append((width, height))
            # writing_data["train"]["path"].append(_img_path) # Put the _img_path var here does not work due to hdf5 not support full rnage numpy 4byte char unicode
            writing_data["train"]["label"].append(label)

            dict_ptr += 1
            #### End of Process the Images ####

            # increase pointer and count
            i += 1
            count += 1
          
      # After the loop, add the last batch to the h5 file
      # Append the last chunk
      # stack
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
      label_dataset.resize(row_count + writing_data["train"]["label"].shape[0], axis=0)
      label_dataset[row_count:] = writing_data["train"]["label"]
      row_count += writing_data["train"]["image"].shape[0]


      ##### Process val folder ########

      # Val writing
      val = f.create_group("val")
      image = val.create_dataset('image', shape=(chunk_size, *img_shape), maxshape=(None, *img_shape),
                                chunks=(chunk_size, *img_shape), dtype=np.uint8)

      resolution = val.create_dataset('resolution', shape=(chunk_size, 2), maxshape=(None, 2),
                                chunks=(chunk_size, 2), dtype=np.uint8)

      # path = val.create_dataset('path', shape=(chunk_size), maxshape=(None,),
      #                           chunks=(chunk_size), dtype=np.byte)

      label_dataset = val.create_dataset('label', shape=(chunk_size, *img_shape), maxshape=(None, *img_shape),
                                chunks=(chunk_size, *img_shape), dtype=np.uint8)
      writing_data = Extractor.initialize_data()
      row_count = 0
      i = 0
      count = 0
      dict_ptr = 0
      for folder_id in range(len(self.val_folder)):
        for eye_id in range(2): # eye 0 is the right eye (which is flipped by mistake in the pipeline), eye 1 is the left eye
          # _path = os.path.join(self.dataset_dir, f"{self.val_folder[i]}", "synthetic")
          # _path_2 = os.path.join(self.dataset_dir, f"{self.val_folder[i]}", "mask-withskin")
          _path = os.path.join(self.dataset_dir, f"{self.val_folder[folder_id]}", "synthetic", f"{eye_id}")
          _path_2 = os.path.join(self.dataset_dir, f"{self.val_folder[folder_id]}", "Mask_withskin", f"{eye_id}")
          for image_name in os.listdir(_path):
            
            if count % chunk_size == 0:
              print(f"batch {i}")
              if count != 0:
                try: 
                  writing_data["val"]["image"] = np.stack(writing_data["val"]["image"], axis=0)
                  writing_data["val"]["resolution"] = np.stack(writing_data["val"]["resolution"], axis=0)
                  # writing_data["val"]["path"] = np.stack(writing_data["val"]["path"], axis=0)
                  writing_data["val"]["label"] = np.stack(writing_data["val"]["label"], axis=0)

                  # Append image
                  image.resize(row_count + writing_data["val"]["image"].shape[0], axis=0)
                  image[row_count:] = writing_data["val"]["image"]
                  resolution.resize(row_count + writing_data["val"]["resolution"].shape[0], axis=0)
                  resolution[row_count:] = writing_data["val"]["resolution"]
                  # path.resize(row_count + writing_data["val"]["path"].shape[0], axis=0)
                  # path[row_count:] = writing_data["val"]["path"]
                  label_dataset.resize(row_count + writing_data["val"]["label"].shape[0], axis=0)
                  label_dataset[row_count:] = writing_data["val"]["label"]
                  row_count += writing_data["val"]["image"].shape[0]
                  writing_data = Extractor.initialize_data()
                except: # ValueError: need at least one array to stack
                  pass

            if dict_ptr % 500 == 0:
              print(f"Extracting val: {dict_ptr}")
            _img_path = os.path.join(_path, image_name)
            _label_path = os.path.join(_path_2, image_name)
            try:
              img = cv2.imread(_img_path, 0) # uint8 0-255
              label: Image = Image.open(_label_path)
              
              if eye_id == 0: # if right eye, then flip both image and label
                img = cv2.rotate(img, cv2.ROTATE_180)
                label = label.rotate(180)

              label: np.ndarray = np.asarray(label) # (480, 640) uint8, [0,255]

            except:
              # Corrupted data file
              print("corrrupt")
              continue
            
            if resizing:
              img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LANCZOS4)
              label = cv2.resize(label, self.image_size)

            img = np.asarray(img) # (240, 320) [0, 255] uint8
            label = np.asarray(label) # (240, 320) 0-1-2-3
            # label = preprocess_label_sparse_np(label) # (480, 640)
            label = preprocess_chengyi_label_sparse_np(label)

            [height, width] = img.shape[:2]

            if (width, height) != self.image_size:
              print("An instance does not match desired image size, skipping")
              continue

            ### Append data
            writing_data["val"]["image"].append(img)
            writing_data["val"]["resolution"].append((width, height))
            # writing_data["val"]["path"].append(_img_path)
            writing_data["val"]["label"].append(label)

            dict_ptr += 1

            i += 1
            count += 1

      try:
        writing_data["val"]["image"] = np.stack(writing_data["val"]["image"], axis=0)
        writing_data["val"]["resolution"] = np.stack(writing_data["val"]["resolution"], axis=0)
        # writing_data["val"]["path"] = np.stack(writing_data["val"]["path"], axis=0)
        writing_data["val"]["label"] = np.stack(writing_data["val"]["label"], axis=0)
        # Append image
        image.resize(row_count + writing_data["val"]["image"].shape[0], axis=0)
        image[row_count:] = writing_data["val"]["image"]
        resolution.resize(row_count + writing_data["val"]["resolution"].shape[0], axis=0)
        resolution[row_count:] = writing_data["val"]["resolution"]
        # path.resize(row_count + writing_data["val"]["path"].shape[0], axis=0)
        # path[row_count:] = writing_data["val"]["path"]
        label_dataset.resize(row_count + writing_data["val"]["label"].shape[0], axis=0)
        label_dataset[row_count:] = writing_data["val"]["label"]
        row_count += writing_data["val"]["image"].shape[0]
        writing_data = Extractor.initialize_data()
      except:
        pass

