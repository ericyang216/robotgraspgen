import pdb
import cv2
import math
import os
import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as T

from robosuite.wrappers import IKWrapper

from sawyer_grasp import SawyerGrasp

from sim_util import *
from grasp_tasks import * 

import torch
from nn.vae import VAEModel
from nn.cnn import CNNModel

RENDER = False
OBS = True
GRIPPER_LENGTH = 0.14
GRIPPER_Y_OFFSET = 0.025
H = 240#480 #240
W = 360##640 #360
CAM_ID = 2 # 0 front view, 1 bird view, 2 agent view
DEPTH_DIR = './data/cube2/depth/'
LABEL_DIR = './data/cube2/label/'

TEST_NAMES = ['box1']
METHOD = 'vae4' #baseline, base+noise, cnn, vae
START = -3.0
END = 3.0

"""
1. Select 5 boxes and positions
2  Run baseline w/ noise many times on each box to get set of successful grasps
3. Evaluate success for 5 items, 10 trials per: baseline, baseline w/ noise, cnn, vae
   Measure the generated target pos and target angle
4. Run models on each of objects and measure the "coverage rate" from (2)

5. Study on VAE latent dimensions? 8, 16, 32, 64 h_dim?
"""

if 'cnn' in METHOD:
    CNN_MODEL_PATH = './checkpoints/cnn_mini_1575429484_900.pt' #cnn_1575397023_900.pt'
    cnn_model = CNNModel()
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH))
    cnn_model.eval()

if 'vae' in METHOD:
    VAE_MODEL_PATH = './checkpoints/vae_mini_1575587840_z4_h16_900.pt'
    #'./checkpoints/vae_mini_1575595272_z128_h16_900.pt'
    #'./checkpoints/vae_mini_1575586849_z64_h16_900.pt'
    #'./checkpoints/vae_mini_1575586455_z32_h16_900.pt'
    #'./checkpoints/vae_mini_1575422961_z16_h16_900.pt'
    #'./checkpoints/vae_mini_1575587244_z8_h16_900.pt'
    #'./checkpoints/vae_mini_1575587840_z4_h16_900.pt'
    #'./checkpoints/vae_mini_1575422961_z16_h16_900.pt'
    vae_model = VAEModel(z_dim=4, h_dim=16)
    vae_model.load_state_dict(torch.load(VAE_MODEL_PATH))
    vae_model.eval()

def baseline_grasp(env, noise=False):
    target_pos, target_angle = env._gen_grasp_gt() #x,y,z,a
    target_angle = T.mat2euler(target_angle)[-1]

    if noise:
        cube_length = env.object.size[0] / 2
        gripper_noise = np.random.uniform(low=0.95, high=1.05)
        lateral_noise = np.random.uniform(low=-0.01, high=0.01)
        longitude_noise = np.random.uniform(low=cube_length*-1.5, high=cube_length)
        angle_noise = np.random.uniform(low=-0.01, high=0.01)
    else:
        gripper_noise = 1
        lateral_noise = 0
        longitude_noise = 0
        angle_noise = 0

    target_pos[2] += GRIPPER_LENGTH * gripper_noise

    target_pos[0] += np.sin(target_angle) * lateral_noise
    target_pos[1] += np.cos(target_angle) * lateral_noise

    target_pos[0] += np.cos(target_angle) * longitude_noise
    target_pos[1] += np.sin(target_angle) * longitude_noise

    target_angle += angle_noise

    return target_pos, target_angle

def nn_grasp(model, depth):
    tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
    output = model(tensor).cpu().detach().numpy()[0]
    target_pos = output[0:3]
    target_angle = np.arctan2(output[3], output[4])
    return target_pos, target_angle

def nn_grasp_vae(model, depth, z=None):
    tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
    output = model(tensor, z_input=z).cpu().detach().numpy()[0]
    target_pos = output[0:3]
    target_angle = np.arctan2(output[3], output[4])
    return target_pos, target_angle

if RENDER == False:
    dummy = SawyerGrasp(
            has_renderer=True,          # no on-screen renderer
            has_offscreen_renderer=False, # off-screen renderer is required for camera observations
            ignore_done=True,            # (optional) never terminates episode
            use_camera_obs=False,         # use camera observations
            camera_height=H,            # set camera height
            camera_width=W,             # set camera width
            camera_name='agentview',     # use "agentview" camera
            use_object_obs=False,        # no object feature when training on pixels
            camera_depth=False,
            target_object='cube'
        )

for TEST_NAME in TEST_NAMES:
    env = SawyerGrasp(
        has_renderer=RENDER,          # no on-screen renderer
        has_offscreen_renderer=OBS, # off-screen renderer is required for camera observations
        ignore_done=True,            # (optional) never terminates episode
        use_camera_obs=OBS,         # use camera observations
        camera_height=H,            # set camera height
        camera_width=W,             # set camera width
        camera_name='agentview',     # use "agentview" camera
        use_object_obs=OBS,        # no object feature when training on pixels
        camera_depth=OBS,
        target_object=TEST_NAME,
        gripper_visualization=True,
        control_freq=100,
    )
    env = IKWrapper(env)

    pos_results = []
    angle_results = []

    for i in range(100):
        zval = START + i * ((END - START) / 100)
        print("Iteration:", i)
        env.reset()
        cam_pos, cam_quat = env._get_cam_pos(CAM_ID)
        cam_pos_new = cam_pos + np.array([0.1, 0.16, 0.3])
        env._set_cam_pos(CAM_ID, cam_pos_new, cam_quat)
        
        reward = 0.

        env.set_robot_joint_positions(np.array([0,-math.pi/2,0,math.pi/2,0,math.pi/2,math.pi/2]))

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

        # cv2.imshow('roi', roi)
        # cv2.waitKey()

        # img = roi / np.max(roi) * 255
        # cv2.imwrite('depth_{}.png'.format(TEST_NAME), img)

        table_top_center, _ = env._get_table_top_center()

        target_pos_gt, target_angle_gt = baseline_grasp(env, noise=False)

        if METHOD == 'base+noise':
            target_pos, target_angle = baseline_grasp(env, noise=True)
        elif 'cnn' in METHOD:
            target_pos, target_angle = nn_grasp(cnn_model, roi) 
        elif 'vae' in METHOD:
            target_pos, target_angle = nn_grasp_vae(vae_model, roi, None)
        else:
            raise NotImplementedError

        pos_results.append(target_pos)
        angle_results.append(target_angle)

    print(target_pos_gt, target_angle_gt)

    print(pos_results)
    print(angle_results)
    print(pos_results[0])

    result_path = './results/zz_interp_' + TEST_NAME + '_' + METHOD + '_%s.npy'
    np.save(result_path % 'pos', np.array(pos_results))
    np.save(result_path % 'angle', np.array(angle_results))