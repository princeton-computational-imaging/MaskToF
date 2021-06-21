import torch
import time
from torch.utils.tensorboard import SummaryWriter
from utils import utils
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from utils.chamfer_distance import ChamferDistance

class Model(object):
    def __init__(self, args, optimizer, net, device, start_iter=0, start_epoch=0, views_x=9, views_y=9, img_height=512, img_width=512):
        self.args = args
        self.optimizer = optimizer
        self.net = net
        self.device = device
        self.num_iter = start_iter
        self.epoch = start_epoch
        self.train_writer = SummaryWriter(self.args.checkpoint_dir)
        self.views_x, self.views_y = views_x, views_y
        self.img_height, self.img_width = img_height, img_width
        self.l1_loss = torch.nn.SmoothL1Loss()
        self.l2_loss = torch.nn.MSELoss()
        self.chamfer_loss = ChamferDistance()
        
        # save initial mask
        utils.check_path(os.path.join(args.checkpoint_dir, "masks/"))
        if self.epoch == 0 and self.epoch in args.mask_checkpoints:
            torch.save(self.net.amplitude_mask.mask, os.path.join(args.checkpoint_dir, "masks/mask_epoch_{0}.pt".format(self.epoch)))
        
    def train(self, train_loader):

        args = self.args
        steps_per_epoch = len(train_loader)
        device = self.device
        self.net.train() # init Module

        # Learning rate summary
        lr = self.optimizer.param_groups[0]["lr"]
        self.train_writer.add_scalar("base_lr", lr, self.epoch + 1)
        
        if self.epoch == self.args.mask_start_epoch:
            # Start gradient
            self.net.amplitude_mask.mask.requires_grad = True

        last_print_time = time.time()


        for i, sample in enumerate(train_loader):
            # increase softmax gamma
            softmax_weight = (1 + (args.softmax_gamma*self.num_iter)**2)
            self.net.amplitude_mask.softmax_weight = softmax_weight
            
            lightfield = sample["lightfield_patches"].to(device)  # [B, num_patches, 81, H, W]
            parameters = sample["parameters"]
            depth = sample["depth_patches"].to(device) # [B, num_patches, 81, H, W]
            depth_gt = sample["depth_gt_patches"].to(device) # [B, num_patches 1, H, W]
            
            # reshape patches -> batches * patches
            lightfield = lightfield.view(-1, *lightfield.shape[2:])
            depth = depth.view(-1, *depth.shape[2:]) # [B * num_patches, 81, H, W]
            depth_gt = depth_gt.view(-1, *depth_gt.shape[2:])
            
            depth_pred = self.net(lightfield, depth, args, parameters, patch=True)
            pcd_pred, pcd_gt = utils.tensor_to_pcd(depth_pred), utils.tensor_to_pcd(depth_gt)
            
            l1_loss = args.l1_weight*self.l1_loss(depth_pred, depth_gt)
            dist1, dist2 = self.chamfer_loss(pcd_pred, pcd_gt)
            chamfer_loss = args.chamfer_weight*(torch.mean(dist1) + torch.mean(dist2))
             
            if args.refinement == "RefinementUnetResnet": # tofnet
                B,C,H,W = depth_pred.shape
                tv_h = torch.pow(depth_pred[:,:,1:,:]-depth_pred[:,:,:-1,:], 2).sum()
                tv_w = torch.pow(depth_pred[:,:,:,1:]-depth_pred[:,:,:,:-1], 2).sum()
                tv_loss = (tv_h + tv_w)/(B*C*H*W)
                loss = F.l1_loss(depth_pred, depth_gt) + 0.0001*tv_loss
                
            else:
                if chamfer_loss != chamfer_loss or self.epoch < self.args.mask_start_epoch: # if is nan
                    loss = l1_loss
                else:
                    loss = l1_loss + chamfer_loss
#             print(depth.mean(), depth_pred.mean())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                        
            # print log
            if self.num_iter % args.print_freq == 0:
                this_cycle = time.time() - last_print_time
                last_print_time += this_cycle
                grad = self.net.amplitude_mask.mask.grad.norm().item() if self.net.amplitude_mask.mask.grad is not None else 0
                print("Epoch: [%3d/%3d] [%5d/%5d] time: %4.2fs l1 loss: %.3f chamfer loss: %.3f grad norm: %.8f" %
                            (self.epoch + 1, args.max_epoch, i + 1, 
                             steps_per_epoch, this_cycle, l1_loss.item(), chamfer_loss.item(),
                             grad))

            if self.num_iter % args.summary_freq == 0:
                img_summary = dict()
                img_summary["lightfield_0"] = lightfield[:,0,:,:]
                img_summary["depth_gt"] = depth_gt
                img_summary["depth_pred"] = depth_pred
                img_summary["absolute_error"] = torch.abs(depth_pred - depth_gt)
                mask = utils.combine_masks(self.net.amplitude_mask.get_mask())
                img_summary["mask_center_256"] = mask[None,None,2176:2432,2176:2432] # [1, 1, H, W]
                img_summary["mask_center_512"] = mask[None,None,2048:2562,2048:2562] # [1, 1, H, W]
                
                utils.save_images(self.train_writer, "train", img_summary, self.num_iter)
                
                img_summary = dict()
                
                self.train_writer.add_scalar("train/loss", loss.item(), self.num_iter)
                self.train_writer.add_scalar("train/l1_loss", l1_loss.item(), self.num_iter)
                self.train_writer.add_scalar("train/chamfer_loss", chamfer_loss.item(), self.num_iter)
#                 self.train_writer.add_scalar("train/softmax_weight", softmax_weight, self.num_iter)
                
            self.num_iter += 1
            
        self.epoch += 1
        
        # save mask if checkpoint
        if self.epoch in args.mask_checkpoints:
            torch.save(self.net.amplitude_mask.mask, os.path.join(args.checkpoint_dir, "masks/mask_epoch_{0}.pt".format(self.epoch)))
        # Always save the latest model for resuming training
        if args.no_validate:
            torch.save(self.net, os.path.join(args.checkpoint_dir, "full_net_latest.pt"))
            utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.net,
                                  epoch=self.epoch, num_iter=self.num_iter,
                                  loss=-1, mask=self.net.amplitude_mask.mask, filename="net_latest.pt")

            # Save checkpoint of specific epoch
            if self.epoch % args.save_ckpt_freq == 0:
                model_dir = os.path.join(args.checkpoint_dir, "models")
                utils.check_path(model_dir)
                utils.save_checkpoint(model_dir, self.optimizer, self.net,
                                      epoch=self.epoch, num_iter=self.num_iter,
                                      loss=-1, mask=self.net.amplitude_mask.mask, save_optimizer=False)

    def validate(self, val_loader):
        args = self.args
        device = self.device
        print("=> Start validation...")

        self.net.eval()

        num_samples = len(val_loader)
        print("=> %d samples found in the validation set" % num_samples)

        val_file = os.path.join(args.checkpoint_dir, "val_results.txt")
        val_loss_chamfer = 0
        val_loss_l1 = 0
        valid_samples = 0

        for i, sample in enumerate(val_loader):
            lightfield = sample["lightfield"].to(device)  # [B, 81, H, W]
            parameters = sample["parameters"]
            depth = sample["depth"].to(device) # [B, 81, H, W]
            depth_gt = sample["depth_gt"].to(device) # [B, 1, H, W]

            valid_samples += 1

            with torch.no_grad():
                depth_pred = self.net(lightfield, depth, args, parameters, patch=False)
                pcd_pred, pcd_gt = utils.tensor_to_pcd(depth_pred), utils.tensor_to_pcd(depth_gt)
            
            val_loss_l1 += args.l1_weight*self.l1_loss(depth_pred, depth_gt)
            dist1, dist2 = self.chamfer_loss(pcd_pred, pcd_gt)
            val_loss_chamfer += args.chamfer_weight*(torch.mean(dist1) + torch.mean(dist2))
            
            # Save 3 images for visualization
            if i in [num_samples // 4, num_samples // 2, num_samples // 4 * 3]:
                img_summary = dict()
                img_summary["lightfield"] = lightfield[:,0,:,:]
                img_summary["depth_gt"] = depth_gt
                img_summary["depth_pred"] = depth_pred
                img_summary["absolute_error"] = torch.abs(depth_pred - depth_gt)
                utils.save_images(self.train_writer, "val" + str(i), img_summary, self.epoch)

        print("=> Validation done!")

        val_loss_chamfer = val_loss_chamfer / valid_samples
        val_loss_l1 = val_loss_l1 / valid_samples
        loss = val_loss_chamfer + val_loss_l1
        # Save validation results
        with open(val_file, "a") as f:
            f.write("epoch: %03d\t" % self.epoch)
            f.write("val_loss_l1: %.3f\t" % val_loss_l1)
            f.write("val_loss_chamfer: %.3f\t" % val_loss_chamfer)

        print("=> Mean validation loss of epoch %d: l1: %.6f chamfer: %.6f" % (self.epoch, val_loss_l1, val_loss_chamfer))
        self.train_writer.add_scalar("val/loss_l1", val_loss_l1, self.num_iter)
        self.train_writer.add_scalar("val/loss_chamfer", val_loss_chamfer, self.num_iter)
        self.train_writer.add_scalar("val/unweighted_loss_l1", val_loss_l1/args.l1_weight, self.num_iter)
        self.train_writer.add_scalar("val/unweighted_loss_chamfer", val_loss_chamfer/args.chamfer_weight, self.num_iter)
        
        # Always save the latest model for resuming training
        torch.save(self.net, os.path.join(args.checkpoint_dir, "full_net_latest.pt"))
        utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.net,
                              epoch=self.epoch, num_iter=self.num_iter,
                              loss=loss, mask=self.net.amplitude_mask.mask, filename="net_latest.pt")

        # Save checkpoint of specific epoch
        if self.epoch % args.save_ckpt_freq == 0:
            model_dir = os.path.join(args.checkpoint_dir, "models")
            utils.check_path(model_dir)
            utils.save_checkpoint(model_dir, self.optimizer, self.net,
                                  epoch=self.epoch, num_iter=self.num_iter,
                                  loss=loss, mask=self.net.amplitude_mask.mask, save_optimizer=False)