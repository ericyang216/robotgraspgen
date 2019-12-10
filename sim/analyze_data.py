import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os

DATA_DIR = "./data/cube2/depth"
LABEL_DIR = "./data/cube2/label"
NUM_DATA = 5011

all_labels = []

for i in range(NUM_DATA):
    data_path = os.path.join(DATA_DIR, "%s.npy" % i)
    label_path = os.path.join(LABEL_DIR, "%s.npy" % i)
    
    label = np.load(label_path)

    pos = label[0:3]
    deg = np.rad2deg(np.arctan2(label[3], label[4]))
    
    shape = np.zeros(3)
    loc = np.zeros(3)
    if label.shape[0] > 5:
        shape = label[5:8]
        loc = label[8:]


    all_labels.append(np.array(list(pos) + [deg] + list(shape) + list(loc)))

    # print(label)
    # print(np.rad2deg(np.arctan2(label[3], label[4])))

    # depth = np.load(data_path)
    # cv2.imshow('depth', depth)
    # cv2.waitKey()

all_labels = np.array(all_labels)

angle_mask = all_labels[:, 3] > -45


plt.scatter(all_labels[:, 0][angle_mask], all_labels[:, 1][angle_mask], c='b', alpha=0.1)
plt.scatter(all_labels[:, 7][angle_mask], all_labels[:, 8][angle_mask], c='r', alpha=0.1)

plt.show()

plt.hist(all_labels[:, 2][angle_mask], alpha=0.5)
plt.show()

plt.hist(all_labels[:, 3][angle_mask], alpha=0.5)
plt.show()