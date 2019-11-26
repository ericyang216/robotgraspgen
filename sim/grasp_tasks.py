import math
import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as T

def init_pose(env):
    table_top_center, _ = env._get_table_top_center()
    target_pos = table_top_center + np.array([0, 0, 0.5])
    move_direction(env, target_pos, up=1)
    move_direction(env, target_pos, right=1, forward=1)
    # point_down(env)

def grasp(env, target_pos, target_angle):
    move_direction(env, target_pos, right=1, forward=1)
    move_direction(env, target_pos, up=1)
    # point_down(env)
    move_direction(env, target_pos, up=1, grasp=0)

    table_top_center, _ = env._get_table_top_center()
    move_direction(env, table_top_center + np.array([0, 0, 0.3]), up=1, grasp=0)

def rotate_gripper(env, z=0):
    current_pos = env._right_hand_pos
    target_rot_X = T.rotation_matrix(0, np.array([1,0,0]), point=current_pos)
    target_rot_Y = T.rotation_matrix(math.pi, np.array([0,1,0]), point=current_pos)
    target_rot_Z = T.rotation_matrix(math.pi, np.array([0,0,1]), point=current_pos)
    target_rot = np.matmul(target_rot_Z, np.matmul(target_rot_Y, target_rot_X))
    target_quat = T.mat2quat(target_rot)

    done_task = False
    while not done_task:
    
        current_pos = env._right_hand_pos
        current_quat = env._right_hand_quat

        # target_rot_Z = T.rotation_matrix(math.pi/180, np.array([0,0,1]), point=current_pos)
        # target_quat = T.mat2quat(target_rot_Z)

        # dquat = target_quat
        
        dquat = T.quat_slerp(current_quat, target_quat, 1)
        dquat = T.quat_multiply(dquat, T.quat_inverse(current_quat))

        if np.abs(dquat[3] - 1.0) < 1e-4:
            done_task = True

        grasp = -1
        dpos = np.zeros(3)
        action = np.concatenate([dpos, dquat, [grasp]])
        obs, reward, done, info = env.step(action)
        env.render()

def point_down(env):
    current_pos = env._right_hand_pos
    target_rot_X = T.rotation_matrix(0, np.array([1,0,0]), point=current_pos)
    target_rot_Y = T.rotation_matrix(math.pi, np.array([0,1,0]), point=current_pos)
    target_rot_Z = T.rotation_matrix(math.pi, np.array([0,0,1]), point=current_pos)
    target_rot = np.matmul(target_rot_Z, np.matmul(target_rot_Y, target_rot_X))
    target_quat = T.mat2quat(target_rot)

    done_task = False
    while not done_task:
    
        current_pos = env._right_hand_pos
        current_quat = env._right_hand_quat
        
        dquat = T.quat_slerp(current_quat, target_quat, 0.01)
        dquat = T.quat_multiply(dquat, T.quat_inverse(current_quat))
        
        if np.abs(dquat[3] - 1.0) < 1e-4:
            done_task = True

        grasp = -1
        dpos = np.zeros(3)
        action = np.concatenate([dpos, dquat, [grasp]])
        obs, reward, done, info = env.step(action)
        env.render()

def move_direction(env, target_pos, up=0, right=0, forward=0, grasp=-1):
    mask = np.array([forward, right, up])
    done_task = False
    while not done_task:
        current_pos = env._right_hand_pos
        table_top_center, _ = env._get_table_top_center()

        dpos = (target_pos - current_pos) * mask

        if np.max(np.abs(dpos)) < 1e-3:
            done_task = True

        dpos = dpos * 0.05
        dquat = np.zeros(4)
        dquat[3] = 1

        action = np.concatenate([dpos, dquat, [grasp]])
        obs, reward, done, info = env.step(action)
        # env.render()