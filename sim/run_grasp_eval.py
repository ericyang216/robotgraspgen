import pdb
import cv2
import math
import os
import argparse
import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as T

from robosuite.wrappers import IKWrapper
from sawyer_grasp import SawyerGrasp

from sim_util import *

import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
mpl.style.use('seaborn-muted')

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

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=int, help='method name')
parser.add_argument('--tests', type=int, action='append', help='test names')
args = parser.parse_args()

TEST_NAMES = args.tests
METHOD = args.method
TRIALS = 50

if 'cnn' in METHOD:
    CNN_MODEL_PATH = './checkpoints2/{}_500.pt'.format(METHOD)
    h_dim = int(METHOD.split('_')[2][1:])
    cnn_model = CNNModel(h_dim=h_dim)
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH))
    cnn_model.eval()

if 'vae' in METHOD:
    VAE_MODEL_PATH = './checkpoints2/{}_500.pt'.format(METHOD)
    z_dim = int(METHOD.split('_')[1][1:])
    h_dim = int(METHOD.split('_')[2][1:])

    vae_model = VAEModel(z_dim=z_dim, h_dim=h_dim)
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

def get_roi(env):
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
    return roi

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

    success_results = []
    pos_results = []
    angle_results = []
    pos_obs_results = []
    orn_obs_results = []

    for i in range(TRIALS):
        print("Iteration:", i)
        env.reset()
        cam_pos, cam_quat = env._get_cam_pos(CAM_ID)
        cam_pos_new = cam_pos + np.array([0.1, 0.16, 0.3])
        env._set_cam_pos(CAM_ID, cam_pos_new, cam_quat)
        
        reward = 0.

        env.set_robot_joint_positions(np.array([0,-math.pi/2,0,math.pi/2,0,math.pi/2,math.pi/2]))

        roi = get_roi(env)
        # img = roi / np.max(roi) * 255
        # cv2.imwrite('depth_{}.png'.format(TEST_NAME), img)

        table_top_center, _ = env._get_table_top_center()

        if METHOD == 'baseline':
            target_pos, target_angle = baseline_grasp(env, noise=False)
        elif METHOD == 'base+noise':
            target_pos, target_angle = baseline_grasp(env, noise=True)
        elif 'cnn' in METHOD:
            target_pos, target_angle = nn_grasp(cnn_model, roi)
        elif 'vae' in METHOD:
            target_pos, target_angle = nn_grasp(vae_model, roi)
        else:
            raise NotImplementedError
    
        # label = np.array([target_pos[0], target_pos[1], target_pos[2], np.sin(target_angle), np.cos(target_angle)])

        pos_traj = []
        angle_traj = []

        for a in range(1000):
            current_pos = env._right_hand_pos

            grasp = -1
            dpos = target_pos + table_top_center - current_pos
            dpos = dpos * np.array([.01, .01, 0])

            current_pos = env._right_hand_pos

            current_quat = env._right_hand_quat

            target_rot_X = T.rotation_matrix(0, np.array([1,0,0]), point=target_pos)
            target_rot_Y = T.rotation_matrix(math.pi, np.array([0,1,0]), point=target_pos)
            target_rot_Z = T.rotation_matrix(math.pi+target_angle, np.array([0,0,1]), point=target_pos)
            target_rot = np.matmul(target_rot_Z, np.matmul(target_rot_Y, target_rot_X))
            target_quat = T.mat2quat(target_rot)

            dquat = T.quat_slerp(current_quat, target_quat, 0.01)
            dquat = T.quat_multiply(dquat, T.quat_inverse(current_quat))

            action = np.concatenate([dpos, dquat, [grasp]])
            obs, reward, done, info = env.step(action)

            pos_traj.append(current_pos - table_top_center)
            angle_traj.append(T.mat2euler(T.quat2mat(current_quat)))
            # env.render()

        time = 0



        done_task = False
        while not done_task:
            time += 1
            if time > 2000:
                break

            current_pos = env._right_hand_pos
            dpos = target_pos + table_top_center - current_pos

            if np.max(np.abs(dpos)) < 1e-2:
                done_task = True

            dpos = dpos * 0.01
            grasp = -1

            current_quat = env._right_hand_quat
            target_rot_X = T.rotation_matrix(0, np.array([1,0,0]), point=target_pos)
            target_rot_Y = T.rotation_matrix(math.pi, np.array([0,1,0]), point=target_pos)
            target_rot_Z = T.rotation_matrix(math.pi+target_angle, np.array([0,0,1]), point=target_pos)
            target_rot = np.matmul(target_rot_Z, np.matmul(target_rot_Y, target_rot_X))
            target_quat = T.mat2quat(target_rot)
            dquat = T.quat_slerp(current_quat, target_quat, 0.01)
            dquat = T.quat_multiply(dquat, T.quat_inverse(current_quat))

            action = np.concatenate([dpos, dquat, [grasp]])
            obs, reward, done, info = env.step(action)
            
            pos_traj.append(current_pos - table_top_center)
            angle_traj.append(T.mat2euler(T.quat2mat(current_quat)))

            if RENDER:
                env.render()


        pos_obs_results.append(env._right_hand_pos - table_top_center)
        orn_obs_results.append(env._right_hand_quat)
        
        pos_results.append(target_pos)
        angle_results.append(target_angle)

        target_pos += np.array([0, 0, 0.1])
        while reward < 0.5:
            time += 1
            if time > 2000:
                break

            current_pos = env._right_hand_pos

            dpos = target_pos + table_top_center - current_pos 

            if np.max(np.abs(dpos)) < 1e-2:
                done_task = True

            dpos = dpos * 0.01
            grasp = 1
            # dquat = np.array([0, 0, 0, 1])
            current_quat = env._right_hand_quat
            target_rot_X = T.rotation_matrix(0, np.array([1,0,0]), point=target_pos)
            target_rot_Y = T.rotation_matrix(math.pi, np.array([0,1,0]), point=target_pos)
            target_rot_Z = T.rotation_matrix(math.pi, np.array([0,0,1]), point=target_pos)
            target_rot = np.matmul(target_rot_Z, np.matmul(target_rot_Y, target_rot_X))
            target_quat = T.mat2quat(target_rot)

            dquat = T.quat_slerp(current_quat, target_quat, 0.01)
            dquat = T.quat_multiply(dquat, T.quat_inverse(current_quat))

            action = np.concatenate([dpos, dquat, [grasp]])
            obs, reward, done, info = env.step(action)
            
            if RENDER:
                env.render()

        success_results.append(reward)

        if reward > 0.5:
            print(i,"SUCCESS")
        else:
            print(i,"FAIL")

        # pos_traj = np.array(pos_traj)
        # angle_traj = np.array(angle_traj)
        # f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex=True)
        # ax1.plot(range(pos_traj.shape[0]), pos_traj[:,0], target_pos[0])
        # ax1.plot(range(pos_traj.shape[0]), [target_pos[0]] * len(pos_traj), c='r')
        # ax1.set_ylabel('x')
        # ax1.get_yaxis().set_label_coords(-0.1,0.5)

        # ax2.plot(range(pos_traj.shape[0]), pos_traj[:,1], target_pos[1])
        # ax2.plot(range(pos_traj.shape[0]), [target_pos[1]] * len(pos_traj), c='r')
        # ax2.set_ylabel('y')
        # ax2.get_yaxis().set_label_coords(-0.1,0.5)

        # ax3.plot(range(pos_traj.shape[0]), pos_traj[:,2], target_pos[2])
        # ax3.plot(range(pos_traj.shape[0]), [target_pos[2]-0.1] * len(pos_traj), c='r')
        # ax3.set_ylabel('z')
        # ax3.get_yaxis().set_label_coords(-0.1,0.5)

        # ax4.plot(range(angle_traj.shape[0]), angle_traj[:,0])
        # ax4.plot(range(angle_traj.shape[0]), [0] * len(angle_traj), c='r')
        # ax4.set_ylabel('theta x')
        # ax4.get_yaxis().set_label_coords(-0.1,0.5)

        # ax5.plot(range(angle_traj.shape[0]), angle_traj[:,1])
        # ax5.plot(range(angle_traj.shape[0]), [np.pi/2] * len(angle_traj), c='r')
        # ax5.set_ylabel('theta y')
        # ax5.get_yaxis().set_label_coords(-0.1,0.5)

        # ax6.plot(range(angle_traj.shape[0]), angle_traj[:,2])
        # ax6.plot(range(angle_traj.shape[0]), [target_angle] * len(angle_traj), c='r')
        # ax6.set_ylabel('theta z')
        # ax6.get_yaxis().set_label_coords(-0.1,0.5)

        # plt.savefig('box1_baseline_control_error.png')
        # plt.show()

    print(success_results)
    print(pos_results)
    print(angle_results)

    result_path = './results2/' + TEST_NAME + '_' + METHOD + '_%s.npy'
    np.save(result_path % 'success', np.array(success_results))
    np.save(result_path % 'pos', np.array(pos_results))
    np.save(result_path % 'angle', np.array(angle_results))
    np.save(result_path % 'pos_obs', np.array(pos_obs_results))
    np.save(result_path % 'orn_obs', np.array(orn_obs_results))