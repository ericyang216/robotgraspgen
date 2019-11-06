import numpy as np
import math
import pymesh
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt

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
        a = math.acos(min([max([AB.dot(AC) / (dAB * dAC), -1.]), 1.]))

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
    face_samples = mesh.faces[ np.random.choice(M, size=n, p=A)]

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

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(PC[:,0], PC[:,1], PC[:,2])
    # ax.set_xlim([-0.5,0.5])
    # ax.set_ylim([-0.5,0.5])
    # ax.set_zlim([-0.5,0.5])
    
    return PC

mesh = pymesh.load_mesh("test_model/models/model_normalized.obj")
print(mesh.vertices.shape, mesh.faces.shape)
# pymesh.save_mesh("test_model.ply", mesh)
# mesh = PyntCloud.from_file("test_model.ply")
# mesh.plot(mesh=True)
# sample_cloud(mesh)
sample_cloud(mesh, n=256)