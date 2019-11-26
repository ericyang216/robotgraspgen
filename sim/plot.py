import numpy as np
from sim_util import *

pc = np.load("pc_cam_A.npy")
plot_pc(pc)
pc = np.load("pc_cam_B.npy")
plot_pc(pc)
pc = np.load("pc_world_A.npy")
plot_pc(pc)
pc = np.load("pc_world_B.npy")
plot_pc(pc)
