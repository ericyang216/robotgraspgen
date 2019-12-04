import numpy as np
import cv2 
import os

DATA_DIR = "./data/cube2/depth"
LABEL_DIR = "./data/cube2/label"

for i in range(10):
    data_path = os.path.join(DATA_DIR, "%s.npy" % i)
    label_path = os.path.join(LABEL_DIR, "%s.npy" % i)
    
    label = np.load(label_path)
    print(label)
    print(np.rad2deg(np.arctan2(label[3], label[4])))

    depth = np.load(data_path)

    # img = depth / np.max(depth) * 255
    # cv2.imwrite('depth_ex_{}.png'.format(i), depth)
    cv2.imshow('depth', depth)
    cv2.waitKey()