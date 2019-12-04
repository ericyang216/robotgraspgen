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

RENDER = False
OBS = True
GRIPPER_LENGTH = 0.14
GRIPPER_Y_OFFSET = 0.025
H = 240#480 #240
W = 360##640 #360
CAM_ID = 2 # 0 front view, 1 bird view, 2 agent view
DEPTH_DIR = './data/cube2/depth/'
LABEL_DIR = './data/cube2/label/'

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
    target_object='cube',
    gripper_visualization=True,
    control_freq=100,
)
env = IKWrapper(env)



count = 0

while True:
    depth_path = os.path.join(DEPTH_DIR, "%s.npy" % count)
    label_path = os.path.join(LABEL_DIR, "%s.npy" % count)
    if (not os.path.exists(depth_path)) and (not os.path.exists(label_path)):
        break

    count += 1

print("Starting at count:", count)

for i in range(1000):
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

    target_pos, target_angle = env._gen_grasp_gt() #x,y,z,a
    target_angle = T.mat2euler(target_angle)[-1]

    gripper_noise = np.random.uniform(low=0.95, high=1.05)
    target_pos[2] += GRIPPER_LENGTH * gripper_noise

    lateral_noise = np.random.uniform(low=-0.01, high=0.01)
    target_pos[0] += np.sin(target_angle) * lateral_noise
    target_pos[1] += np.cos(target_angle) * lateral_noise

    cube_length = env.object.size[0] / 2

    longitude_noise = np.random.uniform(low=cube_length*-1.5, high=cube_length)
    target_pos[0] += np.cos(target_angle) * longitude_noise
    target_pos[1] += np.sin(target_angle) * longitude_noise

    angle_noise = np.random.uniform(low=-0.01, high=0.01)
    target_angle += angle_noise

    #print(gripper_noise)
    #print(lateral_noise)
    #print(longitude_noise)
    #print(angle_noise)

    #print(target_angle)
    table_top_center, _ = env._get_table_top_center()
    # grasp(env, target_pos, target_angle)

    size = env.object.size
    loc = env.env.pose_in_base_from_name('cube')[:3, 3] - table_top_center
    label = np.array([  target_pos[0], 
                        target_pos[1], 
                        target_pos[2], 
                        np.sin(target_angle), 
                        np.cos(target_angle),
                        size[0],
                        size[1],
                        size[2],
                        loc[0],
                        loc[1],
                        loc[2]              ])

    for a in range(1000):
    # while True:
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
        
        if RENDER:
            env.render()


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

    if reward > 0.5:
        print(i,"SUCCESS")
        depth_path = os.path.join(DEPTH_DIR, "%s.npy" % count)
        label_path = os.path.join(LABEL_DIR, "%s.npy" % count)
        print(depth_path)
        print(label_path)
        np.save(depth_path, roi)
        np.save(label_path, label)
        count += 1
    else:
        print(i,"FAIL")