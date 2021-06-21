import sys
import json
import time
import torch
import numpy as np
import torchvision.utils as vutils
from glob import glob
import logging
import optparse
import os


def tensor_to_pcd(arr):
    B, C, H, W = arr.shape
    arr = arr.squeeze().reshape(B,H*W,1)
    x = torch.arange(0,W, device=arr.device)
    x = x[None,:,None].repeat(B,H,1)
    y = torch.arange(0,H, device=arr.device)
    y = y[None,:,None].repeat(B,1,1).repeat_interleave(W, dim=1)
    pcd = torch.cat([x,y,arr], dim=2)
    return pcd

def read_text_lines(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # explicitly set exist_ok when multi-processing

def save_args(args, filename="args.json"):
    args_dict = vars(args)
    check_path(args.checkpoint_dir)
    save_path = os.path.join(args.checkpoint_dir, filename)

    with open(save_path, "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=False)
        
def count_parameters(net):
    num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return num

def save_checkpoint(save_path, optimizer, net, epoch, num_iter,
                    loss, mask, filename=None, save_optimizer=True):
    # Network
    net_state = {
        "epoch": epoch,
        "num_iter": num_iter,
        "loss": loss,
        "state_dict": net.state_dict()
    }
    net_filename = "net_epoch_{:0>3d}.pt".format(epoch) if filename is None else filename
    net_save_path = os.path.join(save_path, net_filename)
    torch.save(net_state, net_save_path)

    mask_name = net_filename.replace("net", "mask")
    mask_save_path = os.path.join(save_path, mask_name)
    torch.save(mask, mask_save_path)
    
    # Optimizer
    if save_optimizer:
        optimizer_state = {
            "epoch": epoch,
            "num_iter": num_iter,
            "loss": loss,
            "state_dict": optimizer.state_dict()
        }
        optimizer_name = net_filename.replace("net", "optimizer")
        optimizer_save_path = os.path.join(save_path, optimizer_name)
        torch.save(optimizer_state, optimizer_save_path)


def load_checkpoint(net, pretrained_path, return_epoch_iter=False, resume=False, no_strict=False):
    if pretrained_path is not None:
        if torch.cuda.is_available():
            state = torch.load(pretrained_path, map_location="cuda")
        else:
            state = torch.load(pretrained_path, map_location="cpu")

        net.load_state_dict(state["state_dict"])  # optimizer has no argument `strict`

        if return_epoch_iter:
            epoch = state["epoch"] if "epoch" in state.keys() else None
            num_iter = state["num_iter"] if "num_iter" in state.keys() else None
            return epoch, num_iter


def resume_latest_ckpt(checkpoint_dir, net, net_name):
    ckpts = sorted(glob(checkpoint_dir + "/" + net_name + "*.pt"))

    if len(ckpts) == 0:
        raise RuntimeError("=> No checkpoint found while resuming training")

    latest_ckpt = ckpts[-1]
    print("=> Resume latest {0} checkpoint: {1}".format(net_name, os.path.basename(latest_ckpt)))
    epoch, num_iter = load_checkpoint(net, latest_ckpt, True, True)

    return epoch, num_iter

def save_images(logger, mode_tag, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)
    for tag, values in images_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            if len(value.shape) == 3:
                value = value[:, np.newaxis, :, :]
            value = value[:1]
            value = torch.from_numpy(value)

            image_name = "{}/{}".format(mode_tag, tag)
            if len(values) > 1:
                image_name = image_name + "_" + str(idx)
            logger.add_image(image_name, vutils.make_grid(value, padding=0, nrow=1, normalize=True, scale_each=True),
                             global_step)
            
def tensor2numpy(var_dict):
    for key, vars in var_dict.items():
        if isinstance(vars, np.ndarray):
            var_dict[key] = vars
        elif isinstance(vars, torch.Tensor):
            var_dict[key] = vars.data.cpu().numpy()
        else:
            raise NotImplementedError("invalid input type for tensor2numpy")

    return var_dict

def get_all_data_folders(base_dir=None):
    if base_dir is None:
        base_dir = os.getcwd()

    data_folders = []
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for category in categories:
        for scene in os.listdir(os.path.join(base_dir, category)):
            data_folder = os.path.join(*[base_dir, category, scene])
            if os.path.isdir(data_folder):
                data_folders.append(data_folder)

    return data_folders


def get_comma_separated_args(option, opt, value, parser):
    values = [v.strip() for v in value.split(",")]
    setattr(parser.values, option.dest, values)


def parse_options():
    parser = optparse.OptionParser()
    parser.add_option("-d", "--date_folder", type="string", action="callback", callback=get_comma_separated_args,
                      dest="data_folders", help="e.g. stratified/dots,test/bedroom")
    options, remainder = parser.parse_args()

    if options.data_folders is None:
        options.data_folders = get_all_data_folders(os.getcwd())
    else:
        options.data_folders = [os.path.abspath("%s") % d for d in options.data_folders]
        for f in options.data_folders:
            print(f)

    return options.data_folders


############################ MASK OPERATIONS ############################
def combine_masks(masks):
    if len(masks.shape) == 3:
        masks = masks.reshape(9,9,*masks.shape[1:])
    C1, C2, H, W = masks.shape
    combined_mask = torch.zeros(C1*H,C2*W, device=masks.device)
    for i in range(C1):
        for j in range(C2):
            combined_mask[i::C1,j::C2] = masks[i,j]#/masks[i,j].norm()
    return combined_mask

def un_combine_masks(combined_mask, shape):
    C1, C2, H, W = shape
    masks = []
    for i in range(C1):
        for j in range(C2):
            masks.append(combined_mask[i::C1,j::C2])
    return torch.stack(masks)

def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel

def gkern_mask(kernel_mean, kernel_sigma, shape=(9,9,64,64)):
    C1, C2, H, W = shape
    mask = np.zeros((C1*H, C2*W))
    assert C1 == C2
    for i in range(H):
        for j in range(W):
            sig = np.random.normal(kernel_mean, kernel_sigma)
            kernel = gkern(C1, sig)
            mask[i*C1:(i+1)*C1, j*C2:(j+1)*C2] = kernel
    return torch.from_numpy(mask)

