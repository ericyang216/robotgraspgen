import subprocess
import os 

H_RANGE = [16, 32, 64]
Z_RANGE = [4, 8]

METHODS = ['baseline', 'base+noise', 'cnn_z0_h%s', 'vae_z%s_h%s', 'gan_z%s_h%s']
METHODS = ['gan_z%s_h%s']
TESTS = ['box1', 'box2', 'box3', 'box4', 'box5']
test_str = ' '.join(['--tests={}'.format(test) for test in TESTS])

for method in METHODS:
    if 'vae' in method:
        for h in H_RANGE:
            for z in Z_RANGE:
                name = method % (z, h)
                command = 'python run_grasp_eval.py --method={} '.format(name) + test_str
                print(command)
                if not os.path.exists('./results2/box1_{}_success.npy'.format(name)):
                    ret = subprocess.run(command, shell=True)

    if 'gan' in method:
        for h in H_RANGE:
            for z in Z_RANGE:
                name = method % (z, h)
                command = 'python run_grasp_eval.py --method={} '.format(name) + test_str
                print(command)
                if not os.path.exists('./results2/box1_{}_success.npy'.format(name)):
                    ret = subprocess.run(command, shell=True)

    elif 'cnn' in method:
        for h in H_RANGE:
            name = method % h
            command = 'python run_grasp_eval.py --method={} '.format(name) + test_str
            print(command)
            if not os.path.exists('./results2/box1_{}_success.npy'.format(name)):
                ret = subprocess.run(command, shell=True)


    else:
        name = method 
        command = 'python run_grasp_eval.py --method={} '.format(name) + test_str
        print(command)
        if not os.path.exists('./results2/box1_{}_success.npy'.format(name)):
            ret = subprocess.run(command, shell=True)

