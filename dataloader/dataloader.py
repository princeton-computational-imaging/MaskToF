from torch.utils.data import Dataset
import os
import numpy as np
from utils.file_io import read_lightfield, read_parameters, read_depth, read_disparity, read_depth_all_view
from utils.utils import read_text_lines

class LightFieldDataset(Dataset):
    def __init__(self,
                 dataset_name='4DLFB',
                 mode='train',
                 transform=None,
                 augmentation=None):
        super(LightFieldDataset, self).__init__()

        self.dataset_name = dataset_name
        self.mode = mode
        self.transform = transform
        self.augmentation = augmentation

        synthetic_dict = {
            'train': 'filenames/4DLFB_train.txt',
            'val': 'filenames/4DLFB_val.txt',
            'test': 'filenames/4DLFB_test.txt'
        }

        dataset_name_dict = {
            '4DLFB': synthetic_dict
        }

        assert dataset_name in dataset_name_dict.keys()
        self.dataset_name = dataset_name
        self.samples = []
        data_filename = dataset_name_dict[dataset_name][mode]
        lines = read_text_lines(data_filename)

        for line in lines:
            splits = line.split()
            data_folder = splits[0]
            sample = dict()
            sample["data_folder"] = data_folder
            sample["lightfield"] = read_lightfield(data_folder)/255. # convert to 0-1 range
            sample["parameters"] = read_parameters(data_folder)
            sample["depth"] = read_depth_all_view(data_folder, N=81)*1000. # convert to mm
            sample["depth_gt"] = read_depth(data_folder)*1000. # convert to mm

            scale = min(1000/np.percentile(sample["depth_gt"], 99.9), 1) # 1000mm max depth
            sample["depth"] = scale * sample["depth"]
            sample["depth_gt"] = scale * sample["depth_gt"]  
            
            if self.transform is not None:
                sample = self.transform(sample)
                
            self.samples.append(sample)

    def __getitem__(self, index):
        sample = self.samples[index]
          
        if self.augmentation is not None:
            sample = self.augmentation(sample)

        return sample

    def __len__(self):
        return len(self.samples)