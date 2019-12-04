import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os

DATA_DIR = "./data/cube2/depth"
LABEL_DIR = "./data/cube2/label"
NUM_DATA = 3000

all_labels = []

for i in range(NUM_DATA):
    data_path = os.path.join(DATA_DIR, "%s.npy" % i)
    label_path = os.path.join(LABEL_DIR, "%s.npy" % i)
    
    label = np.load(label_path)
    deg = np.rad2deg(np.arctan2(label[3], label[4]))
    all_labels.append([label[0], label[1], label[2], deg])

    # print(label)
    # print(np.rad2deg(np.arctan2(label[3], label[4])))

    # depth = np.load(data_path)
    # cv2.imshow('depth', depth)
    # cv2.waitKey()

all_labels = np.array(all_labels)

angle_mask = all_labels[:, 3] > -3


plt.scatter(all_labels[:, 0][angle_mask], all_labels[:, 1][angle_mask], alpha=0.5)
plt.show()

plt.hist(all_labels[:, 2][angle_mask], alpha=0.5)
plt.show()

plt.hist(all_labels[:, 3][angle_mask], alpha=0.5)
plt.show()