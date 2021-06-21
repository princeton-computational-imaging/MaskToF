import torch
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os

from dataloader.dataloader import LightFieldDataset
from dataloader import transforms
from utils import utils
from utils import tof
import model
from nets.MaskToFNet import MaskToFNet

parser = argparse.ArgumentParser()

# Training args
parser.add_argument("--mode", default="train", type=str, help="Network mode [train, val, test]")
parser.add_argument("--checkpoint_dir", default="checkpoints/4DLFB", type=str, required=True, help="Directory to save model checkpoints and logs")
parser.add_argument("--dataset_name", default="4DLFB", type=str, help="Dataset name [4DLFB]")
parser.add_argument('--pretrained_net', default=None, type=str, help='Pretrained network')

parser.add_argument("--batch_size", default=12, type=int, help="Batch size for training")
parser.add_argument("--num_workers", default=2, type=int, help="Number of workers for data loading")
parser.add_argument("--seed", default=42, type=int, help="Seed for PyTorch/NumPy.")
parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay for optimizer")
parser.add_argument("--max_epoch", default=540, type=int, help="Maximum number of epochs for training")
parser.add_argument("--milestones", default="80,160,240,320,400,480", type=str, help="Milestones for MultiStepLR")
parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
parser.add_argument("--no_validate", action="store_true", help="No validation")

parser.add_argument('--print_freq', default=1, type=int, help='Print frequency to screen (# of iterations)')
parser.add_argument('--summary_freq', default=5, type=int, help='Summary frequency to tensorboard (# of iterations)')
parser.add_argument('--val_freq', default=5, type=int, help='Validation frequency (# of epochs)')
parser.add_argument('--save_ckpt_freq', default=10, type=int, help='Save checkpoint frequency (# of epochs)')
parser.add_argument("--mask_checkpoints", default="999999", type=str, help="Epochs at which to save mask to file.  (# of epochs)")

# Image-specific
parser.add_argument('--views_x', default=9, type=int, help='Lightfield dimension 0')
parser.add_argument('--views_y', default=9, type=int, help='Lightfield dimension 1')
parser.add_argument('--img_height', default=512, type=int, help='Sub-aperture view height.')
parser.add_argument('--img_width', default=512, type=int, help='Sub-aperture view width.')
parser.add_argument('--num_patches', default=9, type=int, help='Number of patches for patch-based training.')
parser.add_argument('--patch_height', default=80, type=int, help='Image patch height for patch-based training.')
parser.add_argument('--patch_width', default=80, type=int, help='Image patch width for patch-based training.')
parser.add_argument('--pad_x', default=8, type=int, help='Patch padding in width.')
parser.add_argument('--pad_y', default=8, type=int, help='Patch padding in height.')


# Network-specific
parser.add_argument("--f_list", default="100e6", type=str, help="List of modulation frequencies for phase unwrapping.")
parser.add_argument("--g", default=20, type=float, help="Gain of the sensor. Metric not defined.")
parser.add_argument("--T", default=1000, type=float, help="Integration time. Metric not defined.")
parser.add_argument("--mT", default=2000, type=float, help="Modulation period. Default 2x integration time.")
parser.add_argument("--AWGN_sigma", default=3, type=float, help="Additive white gaussian noise's standard deviation.")
parser.add_argument("--init", default="uniform", type=str, help="Mask initiliazation [ones, zeros, uniform, bernoulli, barcodeX, custom, gaussian_circlesX,Y]")
parser.add_argument("--use_net", action="store_true", help="Add encoder-decoder unet for reconstruction.")
parser.add_argument("--net_lr", default=0.004, type=float, help="Network learning rate")
parser.add_argument("--mask_lr", default=0.1, type=float, help="Mask learning rate")
parser.add_argument("--mask_start_epoch", default=70, type=int, help="Epoch at which to begin updating mask.")
parser.add_argument("--softmax_gamma", default=0, type=float, help="Gamma for mask sigmoid scaling factor. scale = (1 + (gamma*t)^2)")
parser.add_argument("--l1_weight", default=100, type=float, help="Weight for L1 loss term.")
parser.add_argument("--chamfer_weight", default=0.08, type=float, help="Weight for chamfer loss term.")
parser.add_argument("--refinement", default="Refinement", type=str, help="Filename of refinement augmentation (For import).")


args = parser.parse_args()

utils.check_path(args.checkpoint_dir)
utils.save_args(args)

def main():    
    # Seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # speedup if input is same size
    torch.backends.cudnn.benchmark = True
    
    print("=> Training args: {0}".format(args))
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("=> Training on {0} GPU(s)".format(torch.cuda.device_count()))
    else:
        device = torch.device("cpu")
        print("=> Training on CPU")

    # Train loader
    train_transform_list = [transforms.RGBtoNIR(),
                            transforms.ToTensor()
                           ]
    train_augmentation_list = [transforms.ToRandomPatches(args.num_patches, args.patch_width, args.patch_height)]
    train_transform = transforms.Compose(train_transform_list)
    train_augmentation = transforms.Compose(train_augmentation_list)
    train_data = LightFieldDataset(dataset_name=args.dataset_name,
                                              mode=args.mode,
                                              transform=train_transform,
                                              augmentation=train_augmentation)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Validation loader
    val_transform_list = [transforms.RGBtoNIR(),
                          transforms.ToTensor()
                         ]
    val_transform = transforms.Compose(val_transform_list)
    val_data = LightFieldDataset(dataset_name=args.dataset_name,
                                            mode="val",
                                            transform=val_transform)

    val_loader = DataLoader(dataset=val_data, batch_size=4, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    print("=> {} training samples found in the training set".format(len(train_data)))
    
    # Network
    net = MaskToFNet(args, device).to(device)
    print(net.parameters())
    
    net_params = list(filter(lambda kv: kv[0] != "amplitude_mask.mask", net.named_parameters()))
    mask_params = list(filter(lambda kv: kv[0] == "amplitude_mask.mask", net.named_parameters()))
    net_params = [kv[1] for kv in net_params]  # kv is a tuple (key, value)
    mask_params = [kv[1] for kv in mask_params]
    params_group = [{'params': net_params, 'lr': args.net_lr},
                    {'params': mask_params, 'lr': args.mask_lr}, ]    
    optimizer = torch.optim.Adam(params_group, weight_decay=args.weight_decay)

    print("%s" % net)

    if args.pretrained_net is not None:
        logger.info("=> Loading pretrained network: %s" % args.pretrained_net)
        # Enable training from a partially pretrained model
        utils.load_checkpoint(aanet, args.pretrained_net)

    # Parameters
    num_params = utils.count_parameters(net)
    print("=> Number of trainable parameters: %d" % num_params)

    # Resume training
    if args.resume:
        # Load Network
        start_epoch, start_iter = utils.resume_latest_ckpt(args.checkpoint_dir, net, "net")
        # Load Optimizer
        utils.resume_latest_ckpt(args.checkpoint_dir, optimizer, "optimizer")
    else:
        start_epoch = 0
        start_iter = 0

    args.f_list = [float(f) for f in args.f_list.split(",")]
    args.milestones = [int(step) for step in args.milestones.split(",")]
    args.mask_checkpoints = [int(i) for i in args.mask_checkpoints.split(",")]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)
    train_model = model.Model(args, optimizer, net, device, start_iter, start_epoch, 
                              views_x=args.views_x, views_y=args.views_y, img_height=args.img_height, img_width=args.img_width)

    print("=> Start training...")

    for epoch in range(start_epoch, args.max_epoch):
        train_model.train(train_loader)
        if not args.no_validate:
            if epoch % args.val_freq == 0 or epoch == (args.max_epoch - 1):
                train_model.validate(val_loader)
        lr_scheduler.step()

    print("=> End training\n\n")


if __name__ == "__main__":
    main()