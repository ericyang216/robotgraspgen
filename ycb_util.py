import numpy as np
import math
import pymesh
import glob
import os
from tqdm import tqdm
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import h5py as h5
import cv2
import pdb
from open3d import *    


YCB_DATA_DIR = "ycb/ycb/"
YCB_OBJECTS = ["001_chips_can", 
            "002_master_chef_can", 
            "003_cracker_box",
            "004_sugar_box",
            "005_tomato_soup_can",
            "006_mustard_bottle",
            "007_tuna_fish_can",
            "008_pudding_box",
            "009_gelatin_box",
            "010_potted_meat_can",
            "011_banana",
            "012_strawberry",
            "013_apple",
            "014_lemon",
            "015_peach",
            "016_pear",
            "017_orange",
            "018_plum"]

YCB_VIEW_ANGLES = np.linspace(0, 360, num=120).astype(int)

def depth_to_rgb_uv(xyz, rgb_K, T_rgb_depth):
    # Point in RGB cam coordinates
    rgb_p = np.matmul(T_rgb_depth, xyz)

    # Point on RGB cam image
    rgb_uv = np.matmul(rgb_K, rgb_p[0:3])
    rgb_u = int(rgb_uv[0]/rgb_uv[2] + 0.5)
    rgb_v = int(rgb_uv[1]/rgb_uv[2] + 0.5)
    return (rgb_u, rgb_v)

def point_cloud_from_depth(depth_map, depth_K, rgb_K=None, mask=None, T_rgb_depth=None):
    H, W = depth_map.shape
    pc = np.zeros((H * W, 3))
    N = 0

    inv_fx = 1.0 / depth_K[0,0]
    inv_fy = 1.0 / depth_K[1,1]
    cx = depth_K[0,2]
    cy = depth_K[1,2]

    rgb_im = np.zeros_like(mask)
    max_z = np.max(depth_map)

    for v in range(H):
        for u in range(W):
            z = depth_map[v, u]
            if z > 0:

                x = ((u - cx) * z) * inv_fx
                y = ((v - cy) * z) * inv_fy

                # Check if point is in segmentation mask of rgb image
                # if mask is not None:
                #     rgb_H, rgb_W = mask.shape

                #     # Point in RGB cam coordinates
                #     xyz = np.array([x, y, z, 1.])
                #     rgb_u, rgb_v = depth_to_rgb_uv(xyz, rgb_K, T_rgb_depth)

                #     if rgb_u >= rgb_W or rgb_v >= rgb_H or rgb_u < 0 or rgb_v < 0:
                #         continue

                #     rgb_im[rgb_v, rgb_u] = int((z / max_z) * 255)

                if mask is not None:

                    if u >= W or v >= H or u < 0 or v < 0 or mask[v,u] == 255:
                        continue

                    # rgb_im[rgb_v, rgb_u] = int((z / max_z) * 255)

  
                pc[N, 0] = x
                pc[N, 1] = y
                pc[N, 2] = z

                N += 1
    # cv2.imshow('rgb_im', rgb_im)
    # cv2.waitKey()

    pc = np.delete(pc, range(N, H * W), axis=0)        
    return pc

def get_depth_image(target_object, viewpoint_camera, viewpoint_angle):
    basename = "{0}_{1}".format(viewpoint_camera, viewpoint_angle)
    depth_file = os.path.join(YCB_DATA_DIR + target_object, basename + ".h5")
    depth_map = h5.File(depth_file, 'r')["depth"][:]
    # depth_map = filter_discontinuities(depth_map)

    calibration_file = os.path.join(YCB_DATA_DIR + target_object, "calibration.h5")
    calibration = h5.File(calibration_file, 'r')

    rgb_K = calibration["{0}_rgb_K".format(viewpoint_camera)][:]

    depth_K = calibration["{0}_ir_K".format(viewpoint_camera)][:]
    depth_scale = np.array(calibration["{0}_ir_depth_scale".format(viewpoint_camera)]) * .0001 # 100um to meters

    depth_from_ref = "H_{0}_ir_from_{1}".format(viewpoint_camera, "NP5")
    rgb_from_ref = "H_{0}_from_{1}".format(viewpoint_camera, "NP5")

    T_depth_ref = calibration[depth_from_ref][:]
    T_rgb_ref = calibration[rgb_from_ref][:]

    T_rgb_depth = np.matmul(T_rgb_ref, np.linalg.inv(T_depth_ref))

    return depth_map * depth_scale, depth_K, rgb_K, T_rgb_depth

def get_rgb_seg(target_object, viewpoint_camera, viewpoint_angle):
    basename = "masks/{0}_{1}_mask".format(viewpoint_camera, viewpoint_angle)
    mask_file = os.path.join(YCB_DATA_DIR + target_object, basename + ".pbm")
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (640, 480))
    return mask

def filter_and_sample(pc, n=128):
    distance_threshold = 0.25 # meters
    avg = np.mean(pc, axis=0)
    diff = pc - avg[np.newaxis, :]
    dist = np.linalg.norm(diff, axis=1)
    outliers = np.argwhere(dist > distance_threshold)  
    pc_filtered = np.delete(pc, outliers, axis=0)

    sample_idx = np.random.randint(0, high=pc_filtered.shape[0], size=n)
    pc_sampled = pc_filtered[sample_idx]

    pc_centered = pc_sampled - np.mean(pc_sampled, axis=0)

    return pc_centered

def plot_pc(pc, CXYZ=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Torch dimension order: CX
    if CXYZ:
        ax.scatter(pc[0,:], pc[1,:], pc[2,:])
    # Default dimension order: XC   
    else:
        ax.scatter(pc[:,0], pc[:,2], pc[:,1])
    ax.set_xlim([-0.5,0.5])
    ax.set_zlim([-0.5,0.5])
    ax.set_ylim([-0.5,0.5])
    plt.show() 

def pc_from_ply(ply_file, n=1024):
    # pc = read_point_cloud(ply_file) # Read the point cloud
    pc = pymesh.load_mesh(ply_file)
    points = pc.vertices
    sample_idx = np.random.randint(0, high=points.shape[0], size=n)

    points = points[sample_idx]

    x = points[:, 0][:, np.newaxis]
    z = points[:, 1][:, np.newaxis]
    y = points[:, 2][:, np.newaxis]

    return np.concatenate([x, y, z], axis=1)

def generate_gt_pc(target_object, n=1024):
    OBJECT_PATH = 'ycb/ycb/%s/' % target_object
    ply_path = OBJECT_PATH + 'clouds/merged_cloud.ply'

    for i in range(10):
        pc = pc_from_ply(ply_path, n=n)

        save_path = OBJECT_PATH + ('clouds/cloud_%s_%s.npy' % (n, i))
        print(save_path)
        np.save(save_path, pc)

def data_files_exist(target_object, view_angle, CAM):
    basename = "{}_{}".format(CAM, view_angle)
    basedir = os.path.join(YCB_DATA_DIR, target_object)

    calibration = os.path.join(basedir, 'calibration.h5')
    depth = os.path.join(basedir, basename + ".h5")
    mask = os.path.join(basedir, "masks/{}".format(basename + "_mask.pbm"))

    return os.path.exists(calibration) and os.path.exists(depth) and os.path.exists(mask)

def generate_partial_pc(target_object, view_angle, partial_n=128):
    CAM = "NP1"

    if not data_files_exist(target_object, view_angle, CAM):
        print("{} {} {} data does not exist: skipping".format(target_object, CAM, view_angle))
        return

    mask = get_rgb_seg(target_object, CAM, view_angle)
    depth, depth_K, rgb_K, T_rgb_depth = get_depth_image(target_object, CAM, view_angle)
    pc = point_cloud_from_depth(depth, depth_K, rgb_K, mask, T_rgb_depth)
    pc = filter_and_sample(pc, n=partial_n)

    save_path = 'ycb/ycb/%s/clouds/partial_%s_%s_%s.npy' % (target_object, CAM, view_angle, partial_n)
    print(save_path)
    np.save(save_path, pc)

for target_object in YCB_OBJECTS:
    for view_angle in YCB_VIEW_ANGLES:
        generate_partial_pc(target_object, view_angle)

# ply_file = '/home/eric/git/shapegrasp/ycb/ycb/001_chips_can/clouds/merged_cloud.ply'
# pc = pc_from_ply(ply_file)
# plot_pc(pc)

# for target_object in ["001_chips_can"]: #YCB_OBJECTS:
#     for view_angle in YCB_VIEW_ANGLES:
#         print(target_object, view_angle)
#         mask = get_rgb_seg(target_object, "NP1", view_angle)
#         depth, depth_K, rgb_K, T_rgb_depth = get_depth_image(target_object, "NP1", view_angle)
#         pc = point_cloud_from_depth(depth, depth_K, rgb_K, mask, T_rgb_depth)
#         pc = filter_and_sample(pc)

#         print(pc.shape)
#         plot_pc(pc)