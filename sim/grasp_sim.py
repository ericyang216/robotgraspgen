import pdb
import cv2
import math
import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as T

from robosuite.wrappers import IKWrapper

from sawyer_grasp import SawyerGrasp

from sim_util import *

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
    target_object='can'
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

action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
initial_obs = env._get_observation()
object_pos = initial_obs['cube_pos']
print(object_pos)
depth = initial_obs["depth"]

# depth = cv2.flip(depth, 0)
# cv2.imwrite('depth.png', depth / np.max(depth) * 255)
pc = depth_to_pc(depth, K)
np.save("pc_cam.npy", pc)
pc = np.matmul(T.pose_inv(cam_pose), pc.T).T
print(pc.shape)
np.save("pc_world.npy", pc)
exit()
# plot_pc(pc)


target_quat  = env._right_hand_quat
target_pos   = env._get_observation()['cube_pos']

while True:


    current_pos = env._right_hand_pos
    current_quat = env._right_hand_quat

    dpos = (target_pos - current_pos) * 0.05
    drotation = np.eye(3)
    dquat = T.mat2quat(drotation)

    grasp = -1
    action = np.concatenate([dpos, dquat, [grasp]])
    obs, reward, done, info = env.step(action)
    target_pos = obs['cube_pos']

    # print(target_pos)
    # print(current_pos)

    env.render()
    # action = np.zeros(env.dof)
    

# for i in range(1000):
#     action = np.random.randn(env.dof)  # sample random action
#     obs, reward, done, info = env.step(action)  # take action in the environment
#     env.render()  # render on display
#     # print(env._get_observation())