#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

 
# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#



def PCA(points):

    centroid = points.mean(axis=0)
    covariance = (points - centroid).T @ (points - centroid) / points.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    return eigenvalues, eigenvectors



def compute_local_PCA(query_points, cloud_points, radius=0, k=0):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    tree = KDTree(cloud_points)
    if k == 0:
        all_neighbor_indices = tree.query_radius(query_points, radius)
    else:
        all_neighbor_indices = tree.query(query_points, k, return_distance=False)

    all_eigenvalues = []
    all_eigenvectors = []

    for indices in all_neighbor_indices:
        eigenvalues, eigenvectors = PCA(cloud_points[indices, :])
        all_eigenvalues.append(eigenvalues)
        all_eigenvectors.append(eigenvectors)
    
    all_eigenvalues = np.array(all_eigenvalues)
    all_eigenvectors = np.array(all_eigenvectors)

    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius):

    all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius=radius)

    eps = 1e-10
    all_eigenvalues[:,2] += eps

    verticality = 2 * np.arcsin(np.abs(all_eigenvectors[:,2,0])) / np.pi
    linearity = 1 - (all_eigenvalues[:,1] / all_eigenvalues[:,2])
    planarity = (all_eigenvalues[:,1] - all_eigenvalues[:,0]) / all_eigenvalues[:,2]
    sphericity = all_eigenvalues[:,0] / all_eigenvalues[:,2]

    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

		
    # Normal computation
    # ******************
    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, radius=0.50)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('../Lille_street_small_normals_radius.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, k=30)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('../Lille_street_small_normals_knn.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
    

    # BONUS: verticality, linearity, planarity, sphericity
    # ****************************************************
    if True:

        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        verticality, linearity, planarity, sphericity = compute_features(cloud, cloud, radius=0.5)

        write_ply('../Lille_street_small_features.ply', (cloud, verticality, linearity, planarity, sphericity),
                  ['x', 'y', 'z', 'verticality', 'linearity', 'planarity', 'sphericity'])
		
