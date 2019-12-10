import subprocess
import os

H_RANGE = [4, 8, 16, 32, 64]
Z_RANGE = [1, 4, 8, 16, 32, 64]

for h in H_RANGE:
    for z in Z_RANGE:
        command = "python train_vae.py --z_dim={} --h_dim={}".format(z, h)
        print(command)
        if not os.path.exists('./checkpoints2/vae_z{}_h{}_500.pt'.format(z, h)):
            ret = subprocess.run(command, shell=True)

        command = "python train_gan.py --z_dim={} --h_dim={}".format(z, h)
        print(command)
        if not os.path.exists('./checkpoints2/gan_z{}_h{}_500.pt'.format(z, h)):
            ret = subprocess.run(command, shell=True)

    command = "python train_cnn.py --h_dim={}".format(h)
    print(command)
    if not os.path.exists('./checkpoints2/cnn_z0_h{}_500.pt'.format(h)):
        ret = subprocess.run(command, shell=True)
