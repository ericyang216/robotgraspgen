import subprocess

H_RANGE = [4, 8, 16, 32, 64]
Z_RANGE = [1, 4, 8, 16, 32, 64]

METHODS = ['base', 'base+noise', 'cnn_z0_h%s', 'vae_z%s_h%s']
TESTS = ['box1', 'box2', 'box3', 'box4', 'box5']
test_str = ' '.join(['tests={}'.format(test) for test in TESTS])

for method in METHODS:
    if 'vae' in method:
        for h in H_RANGE:
            for z in Z_RANGE:
                name = method % (z, h)
                command = 'python run_grasp_eval.py --method={} '.format(name) + test_str
                print(command)
                ret = subprocess.run(command, shell=True)

    elif 'cnn' in method:
        for h in H_RANGE:
            name = method % h
            command = 'python run_grasp_eval.py --method={} '.format(name) + test_str
            print(command)
            ret = subprocess.run(command, shell=True)
    else:
        name = method 
        command = 'python run_grasp_eval.py --method={} '.format(name) + test_str
        print(command)
        ret = subprocess.run(command, shell=True)

