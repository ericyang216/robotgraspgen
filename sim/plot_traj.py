import numpy as np
import matplotlib.pyplot as plt
import robosuite.utils.transform_utils as T

target_pos = np.array([0.66, 0.16, 0.387])
target_quat = np.array([-9.99048222e-01,4.36193835e-02,6.11740587e-17,2.67091712e-18])

target_angle = T.mat2euler(T.quat2mat(target_quat))
print(target_angle)

pos_traj = np.load('pos_traj.npy')
quat_traj = np.concatenate([np.load('quat_traj.npy'), np.load('quat_traj2.npy')], axis=0)

# plt.plot(range(len(pos_traj)), pos_traj[:,2], label='z')
# plt.plot(range(len(pos_traj)), [target_pos[2]] * len(pos_traj), label='target')
# plt.show()

angle_traj = np.array([T.mat2euler(T.quat2mat(q)) for q in quat_traj])
print(angle_traj[-1])
plt.plot(range(len(angle_traj)), angle_traj[:,2], label='actual')
plt.plot(range(len(angle_traj)), [target_angle[2]] * len(angle_traj), label='target')
plt.xlabel('t')
plt.ylabel('Z-orientation (rad)')
plt.legend()
plt.show()