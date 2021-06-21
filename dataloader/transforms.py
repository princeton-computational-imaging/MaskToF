import torch
import numpy as np
# from PIL import Image
import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    """Convert numpy array to torch tensor"""

    def __call__(self, sample):
        sample["lightfield"] =  torch.from_numpy(sample["lightfield"]) # [9, 9, H, W]
        J,K,H,W = sample["lightfield"].shape
        sample["lightfield"] =  sample["lightfield"].reshape(J*K,H,W)
        
        if "depth_gt" in sample.keys():
            sample["depth_gt"] =  torch.from_numpy(sample["depth_gt"])
            sample["depth_gt"] =  sample["depth_gt"].unsqueeze(0)
        
        if "depth" in sample.keys():
            sample["depth"] =  torch.from_numpy(sample["depth"])
            

        return sample

class RGBtoGray(object):
    """Convert lightfield array to grayscale"""
    
    def __call__(self, sample, R_weight=0.2125, G_weight=0.7154, B_weight=0.0721):
        sample["lightfield"] =  R_weight*sample["lightfield"][...,0] + \
                                G_weight*sample["lightfield"][...,1] + \
                                B_weight*sample["lightfield"][...,2]
        return sample
    
class RGBtoNIR(object):
    """Convert lightfield array to near infra-red"""
    
    def __call__(self, sample):
        interm = np.maximum(sample["lightfield"], 1-sample["lightfield"])[...,::-1]
        nir = (interm[..., 0]*0.229 + interm[..., 1]*0.587 + interm[..., 2]*0.114)**(1/0.25)
        sample["lightfield"] = nir
        return sample
    
class ToRandomPatches(object):
    """Convert full image tensors to random patches"""
    
    def __init__(self, num_patches, patch_width, patch_height, random_rotation=True, random_flip=True):
        self.num_patches = num_patches
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.random_rotation = random_rotation # if true apply 0-3 90 degree rotations
        self.random_flip = random_flip # if true apply random vertical flip
        
    def __call__(self, sample):
        lightfield = sample["lightfield"]
        depth = sample["depth"]
        depth_gt = sample["depth_gt"]
        
        C, H, W = lightfield.shape
        patch_x_coords = torch.randint(0, W - self.patch_width, (1,self.num_patches)).squeeze()
        patch_y_coords = torch.randint(0, H - self.patch_height, (1,self.num_patches)).squeeze()
        
        lightfield_patches = []
        depth_patches = []
        depth_gt_patches = []
    
        for i in range(self.num_patches):
            x1, x2 = patch_x_coords[i], patch_x_coords[i] + self.patch_width
            y1, y2 = patch_y_coords[i], patch_y_coords[i] + self.patch_height
            lightfield_patch = lightfield[:,y1:y2,x1:x2]
            depth_patch = depth[:,y1:y2,x1:x2]
            depth_gt_patch = depth_gt[:,y1:y2,x1:x2]
            if self.random_rotation:
                rot = np.random.randint(0,4)
                lightfield_patch = torch.rot90(lightfield_patch, rot, dims=(1,2))
                depth_patch = torch.rot90(depth_patch, rot, dims=(1,2))
                depth_gt_patch = torch.rot90(depth_gt_patch, rot, dims=(1,2))
            if self.random_flip:
                if np.random.randint(0,2) == 0:
                    lightfield_patch = lightfield_patch.flip(1)
                    depth_patch = depth_patch.flip(1)
                    depth_gt_patch = depth_gt_patch.flip(1)
            
            lightfield_patches.append(lightfield_patch)
            depth_patches.append(depth_patch)
            depth_gt_patches.append(depth_gt_patch)
        
        sample["lightfield_patches"] = torch.stack(lightfield_patches)
        sample["depth_patches"] = torch.stack(depth_patches)
        sample["depth_gt_patches"] = torch.stack(depth_gt_patches)
        
        return sample