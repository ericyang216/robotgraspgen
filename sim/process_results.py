import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os

TEST_NAMES = ['box1', 'box2', 'box3', 'box4', 'box5']
DEFAULT_METHODS = ['baseline', 'base+noise', 'cnn', 'vae16']
ALL_METHODS = ['baseline', 'base+noise', 'cnn', 'vae4', 'vae16', 'vae64']
VAE_METHODS = ['vae4', 'vae16', 'vae64']
RESULTS = ['success', 'pos', 'angle', 'pos_obs', 'orn_obs']
RESULT_DIR = './results/'

mpl.style.use('seaborn-muted')

def bar_chart(test_names, methods, data):
    index = np.arange(len(test_names))
    bar_width = 0.8 / len(methods)

    fig, ax = plt.subplots()

    for j in range(len(methods)):
        ax.bar(index+j*bar_width, data[j], bar_width, label=methods[j])

    ax.set_xticks(index + bar_width + 3*bar_width/2)
    ax.set_xticklabels(test_names)
    ax.legend()

    return fig, ax

def success_rate():
    METHODS = ALL_METHODS #['baseline', 'base+noise', 'cnn', 'vae']

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
    ax.legend(bbox_to_anchor=(1.132, 1), bbox_transform=ax.transAxes)
    plt.show()

def var_pos():
    METHODS = VAE_METHODS #['baseline', 'base+noise', 'cnn', 'vae']

    base_var = []
    for test in TEST_NAMES:
        baseline = np.load(RESULT_DIR + test + '_' + 'base+noise' + '_%s.npy' % 'pos')
        suc = np.load(RESULT_DIR + test + '_' + 'base+noise' + '_%s.npy' % 'success')
        idx = np.argwhere(suc > 0.5)
        if len(idx) == 0:
            var = 0
        else:
            baseline = np.squeeze(baseline[idx, :])
            var = np.var(np.linalg.norm(baseline - np.mean(baseline, axis=0), axis=1))
        base_var.append(var)
    base_var = np.array(base_var)

    data_list = []
    for method in METHODS:
        sub_list = []
        for test in TEST_NAMES:
            sub_list.append(np.load(RESULT_DIR + test + '_' + method + '_%s.npy' % 'pos'))
        data_list.append(sub_list)

    data_list = np.array(data_list)
    avg_pos = np.mean(data_list, axis=3, keepdims=True)
    var = np.var(np.linalg.norm(data_list - avg_pos, axis=3), axis=2) #np.max(np.var(data_list, axis=2), axis=2)

    # var = np.concatenate([base_var[np.newaxis, :], var], axis=0)

    fig, ax = bar_chart(TEST_NAMES, METHODS, var)
    ax.set_ylabel('L2 Position Variance')

    plt.show()

def var_angle():
    METHODS = VAE_METHODS #['baseline', 'base+noise', 'cnn', 'vae']

    base_var = []
    for test in TEST_NAMES:
        baseline = np.load(RESULT_DIR + test + '_' + 'base+noise' + '_%s.npy' % 'angle')
        suc = np.load(RESULT_DIR + test + '_' + 'base+noise' + '_%s.npy' % 'success')
        idx = np.argwhere(suc > 0.5)
        if len(idx) == 0:
            var = 0
        else:
            baseline = baseline[idx]
            var = np.var(baseline)
        base_var.append(var)
    base_var = np.array(base_var)

    data_list = []
    for method in METHODS:
        sub_list = []
        for test in TEST_NAMES:
            sub_list.append(np.load(RESULT_DIR + test + '_' + method + '_%s.npy' % 'angle'))
        data_list.append(sub_list)

    data_list = np.array(data_list)
    var = np.var(data_list, axis=2)
    var = np.concatenate([base_var[np.newaxis, :], var], axis=0)

    fig, ax = bar_chart(TEST_NAMES, ['base+noise'] + METHODS, var)
    ax.set_ylabel('Angle Variance')

    plt.show()

def coverage():
    METHODS = ['baseline', 'cnn', 'vae4', 'vae16', 'vae64']
    METHODS = ['base+noise'] + METHODS
    d = 0.01

    base_list = []
    for test in TEST_NAMES:
        baseline = np.load(RESULT_DIR + test + '_' + 'base+noise' + '_%s.npy' % 'pos')
        suc = np.load(RESULT_DIR + test + '_' + 'base+noise' + '_%s.npy' % 'success')
        idx = np.argwhere(suc > 0.5)
        baseline = baseline[idx, :]
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
            
            if total == 0:
                coverage = 0
            else:
                coverage = count / total
            sub_list.append(coverage)
        data_list.append(sub_list)

    data_list = np.array(data_list)

    fig, ax = bar_chart(TEST_NAMES, METHODS, data_list)
    ax.set_ylabel('Coverage Rate')

    plt.show()
            
def ik_error_pos():
    METHODS = ['baseline', 'base+noise', 'cnn', 'vae']

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

def test_chart():
    METHODS = VAE_METHODS 
    METHODS = ['base+noise'] + METHODS

    data = np.ones([len(METHODS), len(TEST_NAMES)])
    scale = np.array(range(1, len(METHODS)+1))
    data = data / scale[:, np.newaxis]

    fig, ax = bar_chart(TEST_NAMES, METHODS, data)

    ax.set_ylabel('TEST')

    plt.show()

def interp():
    tests = ['box1']
    for test in tests:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        pos = np.load('results/zz_interp_{}_vae64_pos.npy'.format(test))
        angle = np.load('results/zz_interp_{}_vae64_angle.npy'.format(test))
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=40, c=np.arange(pos.shape[0]))

        pos = np.load('results/zz_interp_{}_vae4_pos.npy'.format(test))
        angle = np.load('results/zz_interp_{}_vae4_angle.npy'.format(test))
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=40, c=np.arange(pos.shape[0]))

        gt_pos = np.load('results/{}_baseline_pos.npy'.format(test))[0] - np.array([0, 0, 0.1])
        gt_angle = np.load('results/{}_baseline_angle.npy'.format(test))[0]
        gt_pos = gt_pos[np.newaxis, :]
        gt_pos = np.repeat(gt_pos, 30, axis=0)
        a = np.linspace(-0.03, 0.03, 30)
        gt_pos[:,0] +=  a * np.cos(gt_angle)
        gt_pos[:,1] +=  a * np.sin(gt_angle)
        ax.scatter(gt_pos[:,0], gt_pos[:,1], gt_pos[:,2], s=100, c='r')

        noise_pos = np.load('results/{}_base+noise_pos.npy'.format(test))
        noise_suc = np.load('results/{}_base+noise_success.npy'.format(test))
        noise_pos = noise_pos[noise_suc > -1, :] - np.array([0, 0, 0.1])[np.newaxis, :]
        ax.scatter(noise_pos[:,0], noise_pos[:,1], noise_pos[:,2], s=100, c='k', alpha=0.7)

        cnn_pos = np.load('results/{}_cnn_pos.npy'.format(test)) - np.array([0, 0, 0.1])[np.newaxis, :]
        ax.scatter(cnn_pos[:,0], cnn_pos[:,1], cnn_pos[:,2], s=100, c='b', alpha=1) 

        ax.set_xlim([0.05, 0.14])
        ax.set_ylim([0.15, 0.17])
        ax.set_zlim([0.19, 0.21])
        
        plt.show()

    


##########################################################################################
# success_rate()
# test_chart()
# coverage()
# var_pos()
# var_angle()
interp()
# ik_error_pos()



