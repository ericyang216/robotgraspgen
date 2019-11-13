import numpy as np
import math
import pymesh
import glob
import os
from tqdm import tqdm
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn

def compute_area(vertices, faces):
    """ Area of triangles: A = 0.5 * |AB||AC|sin(a)
    vertices: (N,3) array, 3D coordinate of vertices
    faces:    (M,3) array, 3 vertex indices per face
    
    R:        (M,1) array, area of each face
    """
    M = faces.shape[0]
    R = np.zeros(M)

    for i in range(M):
        A = vertices[faces[i][0]]
        B = vertices[faces[i][1]]
        C = vertices[faces[i][2]]

        AB = B - A
        AC = C - A
        dAB = np.linalg.norm(AB)
        dAC = np.linalg.norm(AC)
    
        # Check for divide by zero if one vector is 0
        if dAB * dAC < 1e-7:
            R[i] = 0.

        else:
            a = np.arccos(min([max([AB.dot(AC) / (dAB * dAC), -1.]), 1.]))
            R[i] = 0.5 * dAC * dAC * np.sin(a)

    return R

def sample_cloud(mesh, n=100):
    """ Sample point cloud on mesh surface.
    mesh: pymesh
    n:    int, number of samples 

    PC:   (n,3) array, point cloud
    """
    PC = np.zeros([n,3])
    M = mesh.faces.shape[0]

    # Sample faces based on area
    A = compute_area(mesh.vertices, mesh.faces)
    A = A / np.sum(A)
    face_samples = mesh.faces[np.random.choice(M, size=n, p=A)]

    # Sample points on faces
    for i in range(n):
        A = mesh.vertices[face_samples[i][0]]
        B = mesh.vertices[face_samples[i][1]]
        C = mesh.vertices[face_samples[i][2]]

        # Uniform sampling of triangle
        # Section 4.2 of https://www.cs.princeton.edu/~funk/tog02.pdf
        (r1, r2) = np.random.uniform(size=2)
        PC[i] = (1 - math.sqrt(r1)) * A +\
                (math.sqrt(r1) * (1 - r2)) * B +\
                (r2 * math.sqrt(r1)) * C

    return PC

def plot_pc(pc, CXYZ=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Torch dimension order: CX
    if CXYZ:
        ax.scatter(pc[0,:], pc[1,:], pc[2,:])
    # Default dimension order: XC   
    else:
        ax.scatter(pc[:,0], pc[:,1], pc[:,2])
    ax.set_xlim([-0.5,0.5])
    ax.set_ylim([-0.5,0.5])
    ax.set_zlim([-0.5,0.5])
    plt.show()

def datalist_pc(n_points=256, paths=None, skip=None):
    MODEL_PATHS = './shapenet/02*/*/models/model_normalized.obj'
    if paths:
        MODEL_PATHS = paths

    for mesh_file in glob.glob(MODEL_PATHS):
        items = mesh_file.split('/')
        synset_id = items[2]
        model_id = items[3] 

        if skip and os.path.exists(skip % (synset_id, model_id)):
            continue

        mesh = pymesh.load_mesh(mesh_file)
        pc = sample_cloud(mesh, n=n_points)

        yield pc, synset_id, model_id

def synset_to_name(offset):
    synset = wn.synset_from_pos_and_offset('n', offset)
    return synset.name().split('.')[0]

def convert_obj_to_stl(obj_file):
    stl_path = obj_file[:-3] + "stl"
    if not os.path.exists(stl_path):
        mesh = pymesh.load_mesh(obj_file)
        pymesh.save_mesh(stl_path, mesh)

model_paths = './shapenet/02*/*/models/model_normalized.obj'
for obj_file in tqdm(glob.glob(model_paths)):
    convert_obj_to_stl(obj_file)

# for pc, syn_id, model_id in datalist_pc(n_points=512, skip="./shapenet/%s/%s/models/model_normalized_512.npy"):
#     save_path = "./shapenet/%s/%s/models/model_normalized_512.npy" % (syn_id, model_id)
#     if os.path.exists(save_path):
#         continue
#     print(save_path)
#     np.save(save_path, pc)
