import subprocess

command = "python grasp_gen.py"

while True:
    ret = subprocess.run(command, shell=True)