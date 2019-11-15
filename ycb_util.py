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

YCB_DATA_DIR = "ycb/ycb/"

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
                if mask is not None:
                    rgb_H, rgb_W = mask.shape

                    # Point in RGB cam coordinates
                    xyz = np.array([x, y, z, 1.])
                    rgb_u, rgb_v = depth_to_rgb_uv(xyz, rgb_K, T_rgb_depth)

                    if rgb_u >= rgb_W or rgb_v >= rgb_H or rgb_u < 0 or rgb_v < 0\
                       or mask[rgb_v, rgb_u] == 255:
                        continue
                    rgb_im[rgb_v, rgb_u] = int((z / max_z) * 255)

    cv2.imshow('rgb_im', rgb_im)
    cv2.waitKey()
    #                 if rgb_u >= rgb_W or rgb_v >= rgb_H or mask[rgb_v, rgb_u] == 255:
    #                     continue

    #             pc[N, 0] = x
    #             pc[N, 1] = y
    #             pc[N, 2] = z

    #             N += 1

    # pc = np.delete(pc, range(N, H * W), axis=0)        
    # return pc

def get_depth_image(target_object, viewpoint_camera, viewpoint_angle):
    basename = "{0}_{1}".format(viewpoint_camera, viewpoint_angle)
    depth_file = os.path.join(YCB_DATA_DIR + target_object, basename + ".h5")
    depth_map = h5.File(depth_file, 'r')["depth"][:]

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
    # unregisteredDepthMap = filterDiscontinuities(unregisteredDepthMap) * depthScale
    return depth_map * depth_scale, depth_K, rgb_K, T_rgb_depth

def get_rgb_seg(target_object, viewpoint_camera, viewpoint_angle):
    basename = "masks/{0}_{1}_mask".format(viewpoint_camera, viewpoint_angle)
    mask_file = os.path.join(YCB_DATA_DIR + target_object, basename + ".pbm")
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (640, 480))
    return mask

def plot_pc(pc, CXYZ=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Torch dimension order: CX
    if CXYZ:
        ax.scatter(pc[0,:], pc[1,:], pc[2,:])
    # Default dimension order: XC   
    else:
        ax.scatter(pc[:,0], pc[:,1], pc[:,2])
    # ax.set_xlim([-0.5,0.5])
    # ax.set_ylim([-0.5,0.5])
    # ax.set_zlim([-0.5,0.5])
    plt.show()   

mask = get_rgb_seg("001_chips_can", "NP1", 0)
depth, depth_K, rgb_K, T_rgb_depth = get_depth_image("001_chips_can", "NP1", 0)
print(rgb_K)
print(depth_K)
pc = point_cloud_from_depth(depth, depth_K, rgb_K, mask, T_rgb_depth)
# # print(pc.shape)
# plot_pc(pc)