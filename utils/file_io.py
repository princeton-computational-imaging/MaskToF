# source: https://github.com/lightfield-analysis/python-tools

import configparser
import os
import sys
import re

import numpy as np


def read_lightfield(data_folder):
    params = read_parameters(data_folder)
    light_field = np.zeros((params["num_cams_x"], params["num_cams_y"], params["height"], params["width"], 3), dtype=np.uint8)

    views = sorted([f for f in os.listdir(data_folder) if f.startswith("input_") and f.endswith(".png")])

    for idx, view in enumerate(views):
        fpath = os.path.join(data_folder, view)
        try:
            img = read_img(fpath)
            light_field[int(idx / params["num_cams_x"]), int(idx % params["num_cams_y"]), :, :, :] = img
        except IOError:
            print("Could not read input file: %s" % fpath)
            sys.exit()

    return light_field


def read_parameters(data_folder):
    params = dict()

    with open(os.path.join(data_folder, "parameters.cfg"), "r") as f:
        parser = configparser.ConfigParser()
        parser.readfp(f)

        section = "intrinsics"
        params["width"] = int(parser.get(section, 'image_resolution_x_px'))
        params["height"] = int(parser.get(section, 'image_resolution_y_px'))
        params["focal_length_mm"] = float(parser.get(section, 'focal_length_mm'))
        params["sensor_size_mm"] = float(parser.get(section, 'sensor_size_mm'))
        params["fstop"] = float(parser.get(section, 'fstop'))

        section = "extrinsics"
        params["num_cams_x"] = int(parser.get(section, 'num_cams_x'))
        params["num_cams_y"] = int(parser.get(section, 'num_cams_y'))
        params["baseline_mm"] = float(parser.get(section, 'baseline_mm'))
        params["focus_distance_m"] = float(parser.get(section, 'focus_distance_m'))
        params["center_cam_x_m"] = float(parser.get(section, 'center_cam_x_m'))
        params["center_cam_y_m"] = float(parser.get(section, 'center_cam_y_m'))
        params["center_cam_z_m"] = float(parser.get(section, 'center_cam_z_m'))
        params["center_cam_rx_rad"] = float(parser.get(section, 'center_cam_rx_rad'))
        params["center_cam_ry_rad"] = float(parser.get(section, 'center_cam_ry_rad'))
        params["center_cam_rz_rad"] = float(parser.get(section, 'center_cam_rz_rad'))

        section = "meta"
        params["disp_min"] = float(parser.get(section, 'disp_min'))
        params["disp_max"] = float(parser.get(section, 'disp_max'))
        params["frustum_disp_min"] = float(parser.get(section, 'frustum_disp_min'))
        params["frustum_disp_max"] = float(parser.get(section, 'frustum_disp_max'))
        params["depth_map_scale"] = float(parser.get(section, 'depth_map_scale'))

        params["scene"] = parser.get(section, 'scene')
        params["category"] = parser.get(section, 'category')
        params["date"] = parser.get(section, 'date')
        params["version"] = parser.get(section, 'version')
        params["authors"] = parser.get(section, 'authors').split(", ")
        params["contact"] = parser.get(section, 'contact')

    return params


def read_depth(data_folder, highres=False):
    fpath = os.path.join(data_folder, "gt_depth_%s.pfm" % ("highres" if highres else "lowres"))
    try:
        data = read_pfm(fpath)
    except IOError:
#         print("Could not read depth file: %s" % fpath)
        return None
    return np.ascontiguousarray(data)


def read_depth_all_view(data_folder, N=81):
    data = []
    for i in range(N):
        fpath = os.path.join(data_folder, "gt_depth_lowres_Cam%03d.pfm" % i)
        try:
            data_i = read_pfm(fpath)
        except IOError:
            print("Could not read depth file: %s" % fpath)
            sys.exit()
        
        data.append( data_i )
     
    data = np.array(data)
    return data

def read_disparity(data_folder, highres=False):
    fpath = os.path.join(data_folder, "gt_disp_%s.pfm" % ("highres" if highres else "lowres"))
    try:
        data = read_pfm(fpath)
    except IOError:
#         print("Could not read disparity file: %s" % fpath)
        return None
    return data

def read_disparity_all_view(data_folder, N=81):
    data = []
    for i in range(N):
        fpath = os.path.join(data_folder, "gt_disp_lowres_Cam%03d.pfm" % i)
        try:
            data_i = read_pfm(fpath)
        except IOError:
            print("Could not read disparity file: %s" % fpath)
            sys.exit()
        data.append( data_i )
    data = np.array(data)
    return data


def read_img(fpath):
    from scipy import misc
    import imageio
    #data = misc.imread(fpath)
    data = imageio.imread(fpath)
    return data


def write_hdf5(data, fpath):
    import h5py
    h = h5py.File(fpath, 'w')
    for key, value in data.iteritems():
        h.create_dataset(key, data=value)
    h.close()


def write_pfm(data, fpath, scale=1, file_identifier="Pf", dtype="float32"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    data = np.flipud(data)
    height, width = np.shape(data)[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder
    print(endianess)

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1

    with open(fpath, 'wb') as file:
        file.write(file_identifier + '\n')
        file.write('%d %d\n' % (width, height))
        file.write('%d\n' % scale)
        file.write(values)

def read_pfm(file):
    file = open(file, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()

    if header.decode("ascii") == 'PF':
        color = True

    elif header.decode("ascii") == 'Pf':
        color = False

    else:
        raise Exception('Not a PFM file.')


    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))

    if dim_match:
        width, height = list(map(int, dim_match.groups()))

    else:
        raise Exception('Malformed PFM header.')
        
    scale = float(file.readline().decode("ascii").rstrip())

    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
        
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    
    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def _get_next_line(f):
    next_line = f.readline().rstrip()
    # ignore comments
    while next_line.startswith('#'):
        next_line = f.readline().rstrip()
    return next_line


