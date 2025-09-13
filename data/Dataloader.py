import json
import os
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from PIL import Image


import h5py


class CustomDataset(Dataset):
    
    def __init__(self, metadata_file, root_dir, transform=None, pre_computed=None, num_classes=24):

        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        self.root_dir = root_dir
        self.transform = transform
        self.pre_computed = pre_computed
        self.h5f = None
        self.num_classes = num_classes
        if self.pre_computed:
            print('Pre-computed features are used.')
            self.h5f = h5py.File(self.pre_computed, 'r')

    def read_data_and_metadata(self, index):
        features = self.h5f['features'][index]
        metadata_str = self.h5f['metadata'][index].decode('utf-8')
        metadata = json.loads(metadata_str) if metadata_str else {}
        
        return features, metadata
    
    def get_metadata(self, idx):
        return self.metadata[idx]
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        if idx >= len(self.metadata):
            raise IndexError("Index out of bound")
        img_name = os.path.join(self.root_dir, os.path.basename(self.metadata[idx]['ground_path']))
        if self.pre_computed:
            features, metadata = self.read_data_and_metadata(idx)
            image = features
            time_str = metadata.get('time')
        else:
            image = Image.open(img_name).convert('RGB')
            time_str = self.metadata[idx]['time']
        hour = datetime.strptime(time_str.split(".")[0], "%Y-%m-%d %H:%M:%S").hour
        hour += datetime.strptime(time_str.split(".")[0], "%Y-%m-%d %H:%M:%S").minute / 60
        class_index = int(hour // (24 / self.num_classes))
        index_tensor = torch.tensor(class_index)
        one_hot_tensor = F.one_hot(index_tensor, num_classes=self.num_classes).squeeze()
        
        if self.transform and not self.pre_computed:
            image = self.transform(image)
        # return image, index_tensor
        return image, one_hot_tensor, hour
