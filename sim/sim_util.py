import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def depth_to_pc(depth, K):
    step = 5
    H, W = depth.shape
    pc = np.ones((H * W, 4))
    N = 0

    inv_fx = 1.0 / K[0,0]
    inv_fy = 1.0 / K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    for v in range(int(H/2-50), int(H/2+50), 2):
        for u in range(int(W/2-50), int(W/2+50), 2):
            z = depth[v, u]
            if z > 0:

                x = ((u - cx) * z) * inv_fx
                y = ((v - cy) * z) * inv_fy

                pc[N, 0] = x 
                pc[N, 1] = y
                pc[N, 2] = -z
                N += 1

    pc = np.delete(pc, range(N, H * W), axis=0)        
    return pc

# Order of [d, w, h]
def plot_pc(pc):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:,1], pc[:,0], pc[:,2])
    # ax.set_xlim([-0.5,0.5])
    # ax.set_zlim([-0.5,0.5])
    # ax.set_ylim([-0.5,0.5])
    plt.show()     