import pdb
import cv2
import math
import os
import gc
import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as T

from robosuite.wrappers import IKWrapper

from sawyer_grasp import SawyerGrasp

from sim_util import *
from grasp_tasks import * 

H = 240#480 #240
W = 360##640 #360

CAM_ID = 2 # 0 front view, 1 bird view, 2 agent view
DEPTH_DIR = './data/cube/depth/'
LABEL_DIR = './data/cube/label/'

# dummy with render to avoid GL init error
dummy = SawyerGrasp(
        has_renderer=True,          # no on-screen renderer
        has_offscreen_renderer=True, # off-screen renderer is required for camera observations
        ignore_done=True,            # (optional) never terminates episode
        use_camera_obs=True,         # use camera observations
        camera_height=H,            # set camera height
        camera_width=W,             # set camera width
        camera_name='agentview',     # use "agentview" camera
        use_object_obs=True,        # no object feature when training on pixels
        camera_depth=True,
        target_object='cube'
    )

for i in range(0, 10):
    depth_path = os.path.join(DEPTH_DIR, "%s.npy" % i)
    label_path = os.path.join(LABEL_DIR, "%s.npy" % i)

    # if os.path.exists(label_path) and os.path.exists(depth_path):
    #     continue

    env = SawyerGrasp(
        has_renderer=True,          # no on-screen renderer
        has_offscreen_renderer=True, # off-screen renderer is required for camera observations
        ignore_done=True,            # (optional) never terminates episode
        use_camera_obs=True,         # use camera observations
        camera_height=H,            # set camera height
        camera_width=W,             # set camera width
        camera_name='agentview',     # use "agentview" camera
        use_object_obs=True,        # no object feature when training on pixels
        camera_depth=True,
        target_object='cube'
    )
    env = IKWrapper(env)

    # reset the environment
    env.reset()

    pos_gt, rot_gt = env._get_grasp_gt()
    quat_gt = T.mat2quat(rot_gt)
    angle_gt = 2 * np.arccos(quat_gt[3])

    label = np.array([pos_gt[0], pos_gt[1], pos_gt[2], np.sin(angle_gt), np.cos(angle_gt), angle_gt])
    print(label)
    # table_top_center, _ = env._get_table_top_center()
    # target_pos = table_top_center + np.array([0, 0, .11]) + pos_gt 

    # offset = np.array([-0.02, 0.02, 0])

    # move out of way of image
    env.set_robot_joint_positions(np.array([-0.28242276,-1.25393477,0.12332545,1.62000231,-0.34764155,1.46593288,2.43873852]))

    obs = env._get_observation()

    depth = obs['depth']
    depth = cv2.flip(depth, 0)

    W_2, H_2 = int(depth.shape[0] / 2), int(depth.shape[0] / 2)
    crop = 64

    start_H = W_2-crop
    end_H = W_2+crop

    start_W = H_2-crop
    end_W = H_2+crop

    roi = depth[start_H:end_H, start_W:end_W]
    # np.save(depth_path, roi)
    # np.save(label_path, label)

    print(depth_path)
    print(label_path)

    # del env
    # gc.collect()
    # cv2.imshow('depth', depth[start_H:end_H, start_W:end_W])

    # cv2.waitKey()
