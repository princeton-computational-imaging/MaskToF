# Mask-ToF  

This is the code for the CVPR 2021 work: [Mask-ToF: Learning Microlens Masks for Flying Pixel Correction in Time-of-Flight Imaging](https://light.princeton.edu/publication/mask-tof/)  

If you use bits and bobs of this code, or find inspiration from it, consider citing the paper:

```
@article{chugunov2021masktof,
title={Mask-ToF: Learning Microlens Masks for Flying Pixel Correction in Time-of-Flight Imaging},
author={Chugunov, Ilya and Baek, Seung-Hwan and Fu, Qiang and Heidrich, Wolfgang and Heide, Felix},
journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2021}
}
```
### Prerequisites:  
* Developed using PyTorch 1.7.0 on Ubuntu x64 machine
* Condensed requirements in `/requirements.txt`
* Full frozen environment can be found in `/conda_env.txt`, but many of these libraries are not necessary to run this code

### Data:  
* Download `test` and `additional` from https://lightfield-analysis.uni-konstanz.de/
* Place samples directly in `/data`
* Confirm that the .txts in `/filenames` match the data structure (update these if you add/reorganize data)
* *Optional:* Download pretrained network checkpoints from [this drive link](https://drive.google.com/file/d/1y6jvOHeZ0483NNbW1-Kks5GZqluS2mZA/view?usp=sharing)
* Extract `pretrained.zip` into `\nets\pretrained`
* *Optional:* Download 'barcode' masks from [this drive link](https://drive.google.com/file/d/1-aYWfIilACarQkAqw6GFSHDw4KpYs-Ry/view?usp=sharing)
* Extract `barcode_masks.zip` into `\utils\barcode_masks`

### Project Structure:
```cpp
MaskToF
  ├── checkpoints  
  │   └── // folder for network checkpoints
  ├── data  
  │   └── // folder for training/test data
  ├── dataloader  
  │   ├── dataloader.py  // pytorch dataloader for lightfields + depth
  │   └── transforms.py  // data augmentations and code to generate image/depth patches
  ├── filenames   
  │   └── // .txt files pointing to data locations
  ├── model.py  // wrapper class for training the network:
  │             // -> load data, calculate loss, print to tensorboard, save network state
  ├── nets  
  │   ├── MaskToFNet.py  // the meat of MaskToF, class for data simulation and learning a mask:
  │   │                  // -> simulate measurements, pass to refinement network, return depth
  │   ├── pretrained
  │   │   └── // folder for pretrained networks
  │   └── refinement  
  │       └── // refinement network architectures
  ├── train.py  // wrapper class for arg parsing and setting up training loop
  └── utils  
      ├── chamfer_distance 
      │   └── // pytorch implementation of chamfer distance metric
      ├── file_io.py  // utils for loading light field data
      ├── tof.py  // utils for simulating time-of-flight measurements
      └── utils.py  // miscellaneous helper functions (e.g. saving network state)
```
### Training:
* Should be as simple as running the debug example with `bash debug.sh`, if all the prerequisite libraries play nice
* Outputs will be saved to `checkpoint_dir=checkpoints/debug/`
* `--init mask_pattern_name` sets the initial mask iterate, in this case `gaussian_circles1.5,0.75`, which is the Gaussian Circle pattern with mean `1.5` and standard distribution `0.75`
* `--use_net` is a flag to jointly train a refinement/reconstruction network, as outlined in the paper
* Additional arguments and descriptions can be found at the top of `/train.py`

### Reconstruction:
The notebook `reconstruction.ipynb` includes an interactive demo for loading a network from a checkpoint folder, visualizing mask structure, simulating amplitude measurements, and performing depth reconstruction.

### Experimental Setup:
If you're building your own experimental prototype, you can reach out to me at `chugunov[at]princeton[dot]edu` for information and advice.

---

Best of luck,  
Ilya
