import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import utils.utils
from utils.tof import *
import importlib


class AmplitudeMask(nn.Module):
    def __init__(self, args, device):
        super(AmplitudeMask, self).__init__()

        if args.init.lower() == "zeros":
            # All Zeroes
            mask = torch.cat([torch.ones((args.views_x*args.views_y, 1, args.patch_height, args.patch_width), device=device),
                              torch.zeros((args.views_x*args.views_y, 1, args.patch_height, args.patch_width), device=device)], dim=1)
        elif args.init.lower() == "ones":
            # All Ones
            mask = torch.cat([torch.zeros((args.views_x*args.views_y, 1, args.patch_height, args.patch_width), device=device),
                              torch.zeros((args.views_x*args.views_y, 1, args.patch_height, args.patch_width), device=device)], dim=1)
        elif args.init.lower() == "uniform":
            # Gaussian Random Mask
            mask = torch.empty(args.views_x*args.views_y, 2, args.patch_height, args.patch_width, device=device).uniform_(0,1)
        elif args.init.lower() == "bernoulli":
            # Bernoulli Random Mask
            mask = torch.empty(args.views_x*args.views_y, 1, args.patch_height, args.patch_width,  device=device).uniform_(0,1)
            mask = torch.bernoulli(mask)
            mask = torch.cat([1 - mask, mask], dim=1)
            
        elif args.init.lower() == "custom":
            # Design your own
            load = torch.tensor([[0.,0,0,0,0,0,0,0,0],
                                 [0,0,0,0,0,0,0,0,0],
                                 [0,0,0,0,1,0,0,0,0],
                                 [0,0,1,0,1,0,1,0,0],
                                 [0,0,1,0,1,0,1,0,0],
                                 [0,0,1,0,1,0,1,0,0],
                                 [0,0,0,0,1,0,0,0,0],
                                 [0,0,0,0,0,0,0,0,0],
                                 [0,0,0,0,0,0,0,0,0]])
            load = load[:,:,None,None]
            m = torch.ones(9,9,args.patch_height,args.patch_width)*load
            m = m.reshape(81,args.patch_height,args.patch_width)
            mask = torch.zeros(81,2,args.patch_height,args.patch_width, device=device)
            mask[:,0,:,:] = 10 - m*10
            mask[:,1,:,:] = m*10
            
        elif "barcode" in args.init.lower() and "/" not in args.init:
            mask = torch.zeros(81,2,512,512, device=device)
            load = torch.from_numpy(np.load("utils/barcode_masks/{0}.npy".format(args.init.lower()))).to(device).reshape(81,512,512)
            mask[:,0,:,:][torch.where(load <= 0)] = 10
            mask[:,1,:,:][torch.where(load > 0)] = 10
            mask = mask[:,:,:args.patch_height,:args.patch_width]
            
        elif "gaussian_circles" in args.init.lower() and "/" not in args.init:
            init = args.init.lower().replace("gaussian_circles","")
            if "," in init:
                mean, sigma = [float(el) for el in init.split(",")]
            else:
                mean, sigma = 1.5, 1
            shape = (args.views_y, args.views_x, args.patch_height, args.patch_width)
            mask = utils.utils.gkern_mask(mean, sigma, shape)
            mask = utils.utils.un_combine_masks(mask, shape)[:,None,:,:]*10 # scale for softmax
            mask = torch.cat([10 - mask, mask], dim=1).float().to(device)
            
        elif "/" in args.init: # load path
            mask = torch.load(args.init, map_location=device)
            
        else:
            raise Exception("Not implemented.")
        
        self.softmax_weight = 1 # softmax temperature
        self.softmax = nn.Softmax(dim=2)
        self.mask = nn.Parameter(data=mask, requires_grad=(args.mask_start_epoch == 0))
        self.mask = self.mask.to(device)
        assert args.img_height % (args.patch_height - args.pad_y*2) == 0
        assert args.img_width % (args.patch_width - args.pad_x*2) == 0
        self.y_repeat = args.img_height//(args.patch_height - args.pad_y*2)
        self.x_repeat = args.img_width//(args.patch_width - args.pad_x*2)
        self.pad_y = args.pad_x
        self.pad_x = args.pad_y
        
    def get_mask_internal(self, patch=True):
        if patch:
            mask = self.mask
        else:
            if self.pad_x > 0 or self.pad_y > 0:
                mask = self.mask[:,:,self.pad_y:-self.pad_y,self.pad_x:-self.pad_x]
            else:
                mask = self.mask
            mask = mask.repeat(1,1,self.y_repeat, self.x_repeat)
        mask = utils.utils.combine_masks(mask[:,1,:,:])[None,None,:,:] # [1,1,9H,9W]
        return mask
   
    def get_mask(self):
        if self.pad_x > 0 or self.pad_y > 0:
            mask = self.mask[:,:,self.pad_y:-self.pad_y,self.pad_x:-self.pad_x]
        else:
            mask = self.mask
        mask = mask.repeat(1,1,self.y_repeat, self.x_repeat)
        return self.softmax(self.softmax_weight * mask)[:,1,:,:]
        
    def forward(self, amplitudes, patch=False):
        if patch:
            mask = self.mask.unsqueeze(0)
        else:
            if self.pad_x > 0 or self.pad_y > 0:
                mask = self.mask[:,:,self.pad_y:-self.pad_y,self.pad_x:-self.pad_x]
            else:
                mask = self.mask
            mask = mask.repeat(1,1,self.y_repeat, self.x_repeat).unsqueeze(0) # [1, C, 2, H, W]
        mask = self.softmax(self.softmax_weight * mask) # threshold 0-1
        mask = mask[:,:,1,:,:] # select 'ON' mask, [B*num_patches, C, H, W]
        mask = mask.unsqueeze(1) # [B*num_patches, 1, C, H, W]
        return mask * amplitudes

class MaskToFNet(nn.Module):
    def __init__(self, args, device):
        super(MaskToFNet, self).__init__()
        self.views_x, self.views_y = args.views_x, args.views_y
        self.img_height, self.img_width = args.img_height, args.img_width
        self.amplitude_mask = AmplitudeMask(args, device)
        
        if args.use_net:
            HourglassRefinement = importlib.import_module('nets.refinement.{0}'.format(args.refinement)).HourglassRefinement
            self.refinement = HourglassRefinement()
    
    def forward(self, lightfield, depth, args, parameters, patch=False):
        B, C, H, W = lightfield.shape
        phi_list = []
        
        # main loop
        for f in args.f_list:
            amplitudes = sim_quad(depth, f, args.T, args.g, lightfield)
            amplitudes = self.amplitude_mask(amplitudes, patch)
            amplitudes = amplitudes.mean(dim=2, dtype=torch.float32) # [B*num_patch, 4, patch_height, patch_width]
            # white gaussian noise
            noise_scale = torch.zeros(amplitudes.shape[0], device=amplitudes.device).uniform_(0.75,1.25)[:,None,None,None] # [B*num_patch, 1,1,1]
            noise = torch.normal(std=args.AWGN_sigma, mean=0, size=amplitudes.shape, 
                                          device=amplitudes.device, dtype=torch.float32)  
            if patch:
                noise = noise * torch.sqrt(noise_scale) # random scale for training
            amplitudes += noise
            phi_est, _, _ = decode_quad(amplitudes, args.T, args.mT)
            phi_list.append(phi_est.squeeze(1))
            
        if len(args.f_list) == 1:
            depth_recon = phase2depth(phi_list[0], args.f_list[0]) # [B, H, W]
        else: # phase unwrapping
            depth_recon = unwrap_ranking(phi_list, args.f_list, min_depth=0, max_depth=6000)
        
        depth_recon = depth_recon.unsqueeze(1) # [B, 1, H, W]
        
        if args.use_net:
            mask = self.amplitude_mask.get_mask_internal(patch=patch)
            depth_recon = self.refinement(depth_recon, mask)
            
        return depth_recon # [B, 1, H, W]

    def process_amplitudes(self, amplitudes, args, patch=False): #phi_est [B, 4, H, W] 
        phi_est, _, _ = decode_quad(amplitudes, args.T, args.mT)
        phi_est = phi_est.squeeze(1)
        depth_recon = phase2depth(phi_est, args.f_list[0]) # [B, H, W]
        depth_recon = depth_recon.unsqueeze(1) # [B, 1, H, W]
        
        if args.use_net:
            mask = self.amplitude_mask.get_mask_internal(patch=patch)
            depth_recon = self.refinement(depth_recon, mask)
            
        return depth_recon # [B, 1, H, W]

    
    def process_depth(self, depth, patch=False):
        mask = self.amplitude_mask.get_mask_internal(patch=patch)
        depth_recon = self.refinement(depth, mask)

        return depth_recon # [B, 1, H, W]
