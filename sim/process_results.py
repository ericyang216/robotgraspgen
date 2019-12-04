import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
import cv2
import os

TEST_NAMES = ['box1', 'box2', 'box3', 'box4', 'box5']
RESULTS = ['success', 'pos', 'angle', 'pos_obs', 'orn_obs']
RESULT_DIR = './results/'

mpl.style.use('seaborn-muted')

def bar_chart(test_names, methods, data):
    index = np.arange(len(test_names))
    bar_width = 0.8 / len(methods)

    fig, ax = plt.subplots()
    for j in range(len(methods)):
        ax.bar(index+j*bar_width, data[j], bar_width, label=methods[j])

    ax.set_xticks(index + bar_width + bar_width/2)
    ax.set_xticklabels(test_names)
    ax.legend()

    return fig, ax

def success_rate():
    METHODS = ['baseline', 'basenoise', 'cnn_mini', 'vae_mini_z16_h16']

    data_list = []
    for method in METHODS:
        sub_list = []
        for test in TEST_NAMES:
            sub_list.append(np.load(RESULT_DIR + test + '_' + method + '_%s.npy' % 'success'))
        data_list.append(sub_list)

    data_list = np.array(data_list)
    success = np.mean(data_list, axis=2)

    fig, ax = bar_chart(TEST_NAMES, METHODS, success)
    ax.set_ylabel('Success Rate')

    plt.show()

def var_pos():
    METHODS = ['baseline', 'basenoise', 'cnn_mini', 'vae_mini_z16_h16']

    data_list = []
    for method in METHODS:
        sub_list = []
        for test in TEST_NAMES:
            sub_list.append(np.load(RESULT_DIR + test + '_' + method + '_%s.npy' % 'pos'))
        data_list.append(sub_list)

    data_list = np.array(data_list)
    var = np.max(np.var(data_list, axis=2), axis=2)

    fig, ax = bar_chart(TEST_NAMES, METHODS, var)
    ax.set_ylabel('Mean Position Variance')

    plt.show()

def var_angle():
    METHODS = ['baseline', 'basenoise', 'cnn_mini', 'vae_mini_z16_h16']

    data_list = []
    for method in METHODS:
        sub_list = []
        for test in TEST_NAMES:
            sub_list.append(np.load(RESULT_DIR + test + '_' + method + '_%s.npy' % 'angle'))
        data_list.append(sub_list)

    data_list = np.array(data_list)
    var = np.var(data_list, axis=2)

    fig, ax = bar_chart(TEST_NAMES, METHODS, var)
    ax.set_ylabel('Angle Variance')

    plt.show()

def coverage():
    METHODS = ['cnn_mini', 'vae_mini_z16_h16']
    d = 0.01

    base_list = []
    for test in TEST_NAMES:
        baseline = np.load(RESULT_DIR + test + '_' + 'basenoise' + '_%s.npy' % 'pos')
        base_list.append(baseline)
    base_list = np.array(base_list)

    pos_list = []
    for method in METHODS:
        sub_list = []
        for test in TEST_NAMES:
            sub_list.append(np.load(RESULT_DIR + test + '_' + method + '_%s.npy' % 'pos'))
        pos_list.append(sub_list)
    pos_list = np.array(pos_list) 

    data_list = []
    for i in range(len(METHODS)):
        sub_list = []

        for j in range(len(TEST_NAMES)):
            total = 0
            count = 0
            for k in range(len(base_list[j])):

                for l in range(pos_list[i][j].shape[0]):

                    if np.linalg.norm(base_list[j][k] - pos_list[i][j][l]) < d:
                        count += 1
                        break

                total += 1
            
            coverage = count / total
            sub_list.append(coverage)
        data_list.append(sub_list)

    data_list = np.array(data_list)

    fig, ax = bar_chart(TEST_NAMES, METHODS, data_list)
    ax.set_ylabel('Coverage Rate')

    plt.show()
            
def ik_error_pos():
    METHODS = ['baseline', 'basenoise', 'cnn', 'vae_z256_h256', 'vae_mini_z256_h256']

    data_list = []
    for method in METHODS:
        sub_list = []
        for test in TEST_NAMES:
            sub_list.append(np.load(RESULT_DIR + test + '_' + method + '_%s.npy' % 'pos'))
        data_list.append(sub_list)

    target_list = np.array(data_list) - np.array([0, 0, 0.1])[np.newaxis, np.newaxis, np.newaxis, :]

    data_list = []
    for method in METHODS:
        sub_list = []
        for test in TEST_NAMES:
            sub_list.append(np.load(RESULT_DIR + test + '_' + method + '_%s.npy' % 'pos_obs'))
        data_list.append(sub_list)

    obs_list = np.array(data_list)

    print(target_list - obs_list)

    error = np.mean(np.linalg.norm(target_list - obs_list, axis=3), axis=2)

    fig, ax = bar_chart(TEST_NAMES, METHODS, error)
    ax.set_ylabel('L2 Error from IK Target Position')

    plt.show()
##########################################################################################
success_rate()
coverage()
var_pos()
var_angle()
ik_error_pos()



