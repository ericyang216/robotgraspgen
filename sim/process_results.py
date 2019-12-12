import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os

TEST_NAMES = ['box1', 'box2', 'box3', 'box4', 'box5']
DEFAULT_METHODS = ['baseline', 'base+noise']
CNN_METHODS = ['cnn_z0_h4', 'cnn_z0_h8', 'cnn_z0_h16', 'cnn_z0_h32', 'cnn_z0_h64']
ALL_METHODS = ['baseline', 'base+noise', 'cnn', 'vae4', 'vae16', 'vae64']
VAE_METHODS = ['vae_z1_h8', 'vae_z2_h8', 'vae_z4_h8', 'vae_z16_h8', 'vae_z32_h8', 'vae_z32_h64']
VAE_METHODS = ['vae_z1_h32', 'vae_z4_h32', 'vae_z8_h32', 'vae_z16_h32', 'vae_z32_h32', 'vae_z64_h32']
VAE_METHODS = ['vae_z4_h8', 'vae_z4_h16', 'vae_z4_h32', 'vae_z4_h64']
VAE_METHODS = ['vae_z64_h8', 'vae_z64_h16', 'vae_z64_h32', 'vae_z64_h64']

BEST_CNN_METHODS = ['cnn_z0_h64']
BEST_VAE_METHODS = ['vae_z32_h16', 'vae_z8_h64']
BEST_GAN_METHODS = ['gan_z4_h64', 'gan_z8_h64']

# GAN_METHODS = ['gan_z4_h32']
RESULTS = ['success', 'pos', 'angle', 'pos_obs', 'orn_obs']
RESULT_DIR = './results2/'

mpl.style.use('seaborn-muted')

def bar_chart(test_names, methods, data):
    index = np.arange(len(test_names))
    bar_width = 0.8 / len(methods)

    fig, ax = plt.subplots()

    for j in range(len(methods)):
        ax.bar(index+j*bar_width, data[j], bar_width, label=methods[j])

    ax.set_xticks(index + bar_width + 4*bar_width/2)
    ax.set_xticklabels(test_names)
    ax.legend()

    return fig, ax

def success_rate():
    METHODS = DEFAULT_METHODS + BEST_CNN_METHODS + BEST_VAE_METHODS + BEST_GAN_METHODS #['baseline', 'base+noise', 'cnn', 'vae']

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
    METHODS = DEFAULT_METHODS + CNN_METHODS + BEST_VAE_METHODS #['baseline', 'base+noise', 'cnn', 'vae']

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
    METHODS = BEST_CNN_METHODS + BEST_VAE_METHODS + BEST_GAN_METHODS #['baseline', 'base+noise', 'cnn', 'vae']
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

def plot_3d(test='box1'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    gt_pos = np.load('results2/{}_baseline_pos.npy'.format(test))[0] - np.array([0, 0, 0.1])
    gt_angle = np.load('results2/{}_baseline_angle.npy'.format(test))[0]
    gt_pos = gt_pos[np.newaxis, :]
    gt_pos = np.repeat(gt_pos, 30, axis=0)
    a = np.linspace(-0.06, 0.06, 30)
    gt_pos[:,0] +=  a * np.cos(gt_angle)
    gt_pos[:,1] +=  a * np.sin(gt_angle)
    ax.scatter(gt_pos[:,0], gt_pos[:,1], gt_pos[:,2], s=100, c='r')

    noise_pos = np.load('results2/{}_base+noise_pos.npy'.format(test))
    noise_suc = np.load('results2/{}_base+noise_success.npy'.format(test))
    noise_pos = noise_pos[noise_suc > 0.5, :] - np.array([0, 0, 0.1])[np.newaxis, :]
    ax.scatter(noise_pos[:,0], noise_pos[:,1], noise_pos[:,2], s=100, c='k', alpha=0.7)
    noise_avg = np.mean(noise_pos, axis=0)
    noise_var = np.std(noise_pos, axis=0)
    # ax.scatter(noise_avg[0], noise_avg[1], noise_avg[2], s=1000, c='k', alpha=0.2)

    # cnn_pos = np.load('results2/{}_cnn_z0_h64_pos.npy'.format(test)) - np.array([0, 0, 0.1])[np.newaxis, :]
    # ax.scatter(cnn_pos[:,0], cnn_pos[:,1], cnn_pos[:,2], s=100, c='b', alpha=1) 

    pos = np.load('results2/{}_gan_z8_h64_pos.npy'.format(test)) - np.array([0, 0, 0.1])[np.newaxis, :]
    # # angle = np.load('results/{}_vae64_angle.npy'.format(test))
    pos_avg = np.mean(pos, axis=0)
    pos_var = np.std(pos, axis=0)
    pos = (noise_var / pos_var) * (pos - pos_avg) + pos_avg
    pos += noise_avg - pos_avg
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=50, c='y')#c=np.arange(pos.shape[0]))
    print(compute_coverage(noise_pos, pos))

    pos = np.load('results2/{}_gan_z4_h64_pos.npy'.format(test)) - np.array([0, 0, 0.1])[np.newaxis, :]
    pos_avg = np.mean(pos, axis=0)
    pos_var = np.std(pos, axis=0)
    pos = (noise_var / pos_var) * (pos - pos_avg) + pos_avg
    pos += noise_avg - pos_avg
    # angle = np.load('results/{}_vae4_angle.npy'.format(test))
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=50, c='g') #np.arange(pos.shape[0]))
    print(compute_coverage(noise_pos, pos))
    ax.set_xlim([0.00, 0.2])
    ax.set_ylim([0.12, 0.22])
    ax.set_zlim([0.21, 0.24])
    
    plt.show()

def compute_coverage(baseline, pos):
    total_coverage_count = 0
    total_coverage = 0
    d = 0.01
    for k in range(len(baseline)):
        for l in range(len(pos)):

            if np.linalg.norm(baseline[k] - pos[l]) < d:
                total_coverage_count += 1
                break

        total_coverage += 1
    return total_coverage_count/total_coverage

def avg_success_coverage():
    # H_RANGE = [16, 32, 64]
    # Z_RANGE = [1, 4, 8, 16, 32]

    H_RANGE = [16, 32, 64]
    Z_RANGE = [4, 8]

    # method = 'vae_z%s_h%s'
    method = 'gan_z%s_h%s'
    for z in Z_RANGE:
        for h in H_RANGE:
            total_trials = 0
            total_success = 0
            total_coverage_count = 0
            total_coverage = 0
            for test in TEST_NAMES:
                name = method % (z, h)

                success = np.load('./results2/{}_{}_success.npy'.format(test, name))
                pos = np.load('./results2/{}_{}_pos.npy'.format(test, name))

                basenoise_success = np.load('./results2/{}_base+noise_success.npy'.format(test))
                basenoise_pos = np.load('./results2/{}_base+noise_pos.npy'.format(test))

                idx = np.argwhere(basenoise_success > 0.5)
                basenoise_pos = basenoise_pos[idx, :]
                d = 0.01
                for k in range(len(basenoise_pos)):
                    for l in range(len(pos)):

                        if np.linalg.norm(basenoise_pos[k] - pos[l]) < d:
                            total_coverage_count += 1
                            break

                    total_coverage += 1
                
                total_success += np.sum(success)
                total_trials += len(success)
                # print(total_success)
                # print(total_trials)

            # print("{0:.2f}".format(total_success/total_trials), end=' & ')
            print("{0:.2f}".format(total_coverage_count/total_coverage), end=' & ')

        print("")



##########################################################################################
# plot_3d(test='box4')
# avg_success_coverage()
success_rate()
coverage()
# var_pos()
# var_angle()
# test_chart()
# interp()
# ik_error_pos()



