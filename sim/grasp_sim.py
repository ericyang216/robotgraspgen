import pdb
import cv2
import math
import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as T

from robosuite.wrappers import IKWrapper

from sawyer_grasp import SawyerGrasp

from sim_util import *
from grasp_tasks import * 

H = 480
W = 640
CAM_ID = 2 # 0 front view, 1 bird view, 2 agent view

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
    target_object='cereal'
)
env = IKWrapper(env)

# reset the environment
env.reset()

f, cx, cy = env._get_cam_K(CAM_ID)
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
print(K)

# Get original camera location
cam_pos, cam_quat = env._get_cam_pos(CAM_ID)
cam_rot = T.quat2mat(cam_quat)
cam_pose = np.eye(4)
cam_pose[:3,:3] = cam_rot
cam_pose[:3, 3] = cam_pos
print(cam_pose)

# Set new camera location
table_pos, table_quat = env._get_body_pos("table")
print("table:", table_pos)
cam_pose_new = T.rotation_matrix(math.pi/2, np.array([0,0,1]), table_pos)
cam_pose_new = np.matmul(cam_pose_new, cam_pose)
cam_quat_new = T.mat2quat(cam_pose_new[:3,:3])
cam_pos_new = cam_pose_new[:3, 3]
env._set_cam_pos(CAM_ID, cam_pos_new, cam_quat_new)
print(cam_pose_new)

cam_pos, cam_quat = env._get_cam_pos(CAM_ID)
cam_rot = T.quat2mat(cam_quat)
cam_pose = np.eye(4)
cam_pose[:3,:3] = cam_rot
cam_pose[:3, 3] = cam_pos
print(cam_pose)

assert np.allclose(cam_pose, cam_pose_new, rtol=1e-05, atol=1e-05)

initial_obs = env._get_observation()
object_pos = initial_obs['cube_pos']
depth = initial_obs['depth']

pc = depth_to_pc(depth, K)
np.save("pc_cam_A.npy", pc)
pc = np.matmul(T.pose_inv(cam_pose), pc.T).T
print(pc.shape)
np.save("pc_world_A.npy", pc)

# Set new camera location
table_pos, table_quat = env._get_body_pos("table")
print("table:", table_pos)
cam_pose_new = T.rotation_matrix(math.pi/2, np.array([0,0,1]), table_pos)
cam_pose_new = np.matmul(cam_pose_new, cam_pose)
cam_quat_new = T.mat2quat(cam_pose_new[:3,:3])
cam_pos_new = cam_pose_new[:3, 3]
env._set_cam_pos(CAM_ID, cam_pos_new, cam_quat_new)
print(cam_pose_new)

cam_pos, cam_quat = env._get_cam_pos(CAM_ID)
cam_rot = T.quat2mat(cam_quat)
cam_pose = np.eye(4)
cam_pose[:3,:3] = cam_rot
cam_pose[:3, 3] = cam_pos
print(cam_pose)

assert np.allclose(cam_pose, cam_pose_new, rtol=1e-05, atol=1e-05)

initial_obs = env._get_observation()
object_pos = initial_obs['cube_pos']
depth = initial_obs['depth']

pc = depth_to_pc(depth, K)
np.save("pc_cam_B.npy", pc)
pc = np.matmul(T.pose_inv(cam_pose), pc.T).T
print(pc.shape)
np.save("pc_world_B.npy", pc)


init_pose(env)
print(env._get_observation()['cube_pos'])
target_grasp = env._get_target_grasp() #x,y,z,a
grasp(env, target_grasp)

# target_quat  = env._right_hand_quat
# target_pos   = 
# grasp = -1
# # init_qpos: array([ 0.    , -1.18  ,  0.    ,  2.18  ,  0.    ,  0.57  ,  3.3161])
# # Move to above table
# target_pos = env._right_hand_pos + np.array([0, 0, 0.2])
# done_task = False
# while not done_task:
#     current_pos = env._right_hand_pos
#     dpos = (target_pos - current_pos) * 0.01
#     if np.max(np.abs(dpos)) < 1e-3:
#         dpos = np.zeros(3)
#         done_task = True

#     drotation = np.eye(3)
#     dquat = T.mat2quat(drotation)
#     action = np.concatenate([dpos, dquat, [grasp]])
#     obs, reward, done, info = env.step(action)
#     env.render()

# target_pos, target_quat = env._get_target_grasp()
# # Move to above object
# done_task = False
# while not done_task:
# # for i in range(1000):

#     current_pos = env._right_hand_pos
#     current_quat = env._right_hand_quat

#     dpos = (target_pos - current_pos) * 0.01
#     drotation = np.eye(3)
#     dquat = T.mat2quat(drotation)

#     if np.max(np.abs(dpos)) < 1e-3:
#         print("GRASP")
#         grasp = 1
#         dpos = np.zeros(3)
#         done_task = True

#     action = np.concatenate([dpos, dquat, [grasp]])
#     obs, reward, done, info = env.step(action)
#     # target_pos = obs['cube_pos']

#     env.render()
#     # action = np.zeros(env.dof)
    
# # Move to above table
# target_pos = env._right_hand_pos + np.array([0, 0, 0.2])
# for i in range(100):
#     current_pos = env._right_hand_pos
#     dpos = (table_pos - current_pos) * 0.01
#     if np.max(np.abs(dpos)) < 1e-3:
#         dpos = np.zeros(3)
#     drotation = np.eye(3)
#     dquat = T.mat2quat(drotation)
#     action = np.concatenate([dpos, dquat, [grasp]])
#     obs, reward, done, info = env.step(action)
#     env.render()

# for i in range(1000):
#     action = np.random.randn(env.dof)  # sample random action
#     obs, reward, done, info = env.step(action)  # take action in the environment
#     env.render()  # render on display
#     # print(env._get_observation())