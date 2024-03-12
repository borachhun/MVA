#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Plane detection with RANSAC
#
#------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time



#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points):
    
    # TODO:
    point_plane = points[0].reshape(-1,1)
    normal_plane = np.cross(points[1] - points[0], points[2] - points[0]).reshape(-1,1)
    normal_plane /= np.linalg.norm(normal_plane)
    
    return point_plane, normal_plane



def in_plane(points, pt_plane, normal_plane, threshold_in=0.1):
    
    # TODO:
    indexes = np.abs(
        (points - pt_plane.squeeze()) @ normal_plane
    ).squeeze() <= threshold_in
        
    return indexes



def RANSAC(points, nb_draws=100, threshold_in=0.1):
    
    best_vote = 3
    best_pt_plane = np.zeros((3,1))
    best_normal_plane = np.zeros((3,1))
    
    # TODO:
    for _ in range(nb_draws):
        point_plane, normal_plane = compute_plane(points[
            np.random.choice(range(points.shape[0]), 3, replace=False)
        ])
        vote = np.sum(in_plane(points, point_plane, normal_plane, threshold_in))
        if vote > best_vote:
            best_vote = vote
            best_pt_plane = point_plane
            best_normal_plane = normal_plane
                
    return best_pt_plane, best_normal_plane, best_vote


def recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2):
    
    nb_points = len(points)
    plane_inds = np.arange(0,0)
    plane_labels = np.arange(0,0)
    remaining_inds = np.arange(0,nb_points)
	
    # TODO:
    for l in range(nb_planes):
        point_plane, normal_plane, _ = RANSAC(points[remaining_inds], nb_draws, threshold_in)
        inds_in_plane = in_plane(points[remaining_inds], point_plane, normal_plane, threshold_in)
        plane_inds = np.append(plane_inds, remaining_inds[inds_in_plane])
        plane_labels = np.append(plane_labels, [l] * len(remaining_inds[inds_in_plane]))
        remaining_inds = remaining_inds[np.logical_not(inds_in_plane)]
    
    return plane_inds, remaining_inds, plane_labels


##### FOR QUESTION 4 #####

from sklearn.neighbors import KDTree


# From TP3
def PCA(points):
    centroid = points.mean(axis=0)
    covariance = (points - centroid).T @ (points - centroid) / points.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvalues, eigenvectors


# From TP3
def compute_normal(points, radius=0, k=0):
    normals = np.zeros((points.shape[0], 3))

    tree = KDTree(points)

    # Query tree for neighbors of points a batch
    # at a time due to memory constraint
    batch_size = 10000

    for i in range(0, points.shape[0], batch_size):
        end_idx = i+batch_size if i+batch_size <= points.shape[0] else points.shape[0]

        if k == 0:
            batch_neighbor_indices = tree.query_radius(points[i:end_idx], radius)
        else:
            batch_neighbor_indices = tree.query(points[i:end_idx], k, return_distance=False)

        for j in range(i, end_idx):
            _, eigenvectors = PCA(points[batch_neighbor_indices[j-i], :])
            normals[j,:] = eigenvectors[:,0]

    return normals


def in_plane_with_normal(points, normals, pt_plane, normal_plane, threshold_in=0.1, threshold_angle=0.087):
    
    indexes = np.logical_and(
        np.abs(
            (points - pt_plane.squeeze()) @ normal_plane
        ).squeeze() <= threshold_in,
        np.arccos(
            np.clip(normals @ normal_plane, -1, 1)
        ).squeeze() <= threshold_angle
    )
        
    return indexes


def RANSAC_with_normal(points, normals, nb_draws=100, threshold_in=0.1, threshold_angle=0.087):
    
    best_vote = 3
    best_pt_plane = np.zeros((3,1))
    best_normal_plane = np.zeros((3,1))
    
    for _ in range(nb_draws):
        point_plane, normal_plane = compute_plane(points[
            np.random.choice(range(points.shape[0]), 3, replace=False)
        ])
        vote = np.sum(in_plane_with_normal(points, normals, point_plane, normal_plane, threshold_in, threshold_angle))
        if vote > best_vote:
            best_vote = vote
            best_pt_plane = point_plane
            best_normal_plane = normal_plane
                
    return best_pt_plane, best_normal_plane, best_vote


def recursive_RANSAC_with_normal(points, normals, nb_draws=100, threshold_in=0.1, threshold_angle=0.087, nb_planes=2):
    
    nb_points = len(points)
    plane_inds = np.arange(0,0)
    plane_labels = np.arange(0,0)
    remaining_inds = np.arange(0,nb_points)
	
    for l in range(nb_planes):
        point_plane, normal_plane, _ = RANSAC_with_normal(points[remaining_inds], normals[remaining_inds], nb_draws, threshold_in, threshold_angle)
        inds_in_plane = in_plane_with_normal(points[remaining_inds], normals[remaining_inds], point_plane, normal_plane, threshold_in, threshold_angle)
        plane_inds = np.append(plane_inds, remaining_inds[inds_in_plane])
        plane_labels = np.append(plane_labels, [l] * len(remaining_inds[inds_in_plane]))
        remaining_inds = remaining_inds[np.logical_not(inds_in_plane)]
    
    return plane_inds, remaining_inds, plane_labels

##########################



#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':


    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']
    nb_points = len(points)
    

    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #
    
    print('\n--- 1) and 2) ---\n')
    
    # Define parameter
    threshold_in = 0.10

    # Take randomly three points
    pts = points[np.random.randint(0, nb_points, size=3)]
    
    # Computes the plane passing through the 3 points
    t0 = time.time()
    pt_plane, normal_plane = compute_plane(pts)
    t1 = time.time()
    print('plane computation done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    t0 = time.time()
    points_in_plane = in_plane(points, pt_plane, normal_plane, threshold_in)
    t1 = time.time()
    print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save extracted plane and remaining points
    write_ply('../plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Computes the best plane fitting the point cloud
    # ***********************************
    #
    #
    
    print('\n--- 3) ---\n')

    # Define parameters of RANSAC
    nb_draws = 100
    threshold_in = 0.10

    # Find best plane by RANSAC
    t0 = time.time()
    best_pt_plane, best_normal_plane, best_vote = RANSAC(points, nb_draws, threshold_in)
    t1 = time.time()
    print('RANSAC done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    points_in_plane = in_plane(points, best_pt_plane, best_normal_plane, threshold_in)
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save the best extracted plane and remaining points
    write_ply('../best_plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_best_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    
    print('\n--- 4) ---\n')
    
    # Define parameters of recursive_RANSAC
    nb_draws = 100
    threshold_in = 0.10
    nb_planes = 2
    
    # Recursively find best plane by RANSAC
    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, nb_draws, threshold_in, nb_planes)
    t1 = time.time()
    print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))
                
    # Save the best planes and remaining points
    write_ply('../best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('../remaining_points_best_planes.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    
    # QUESTION 4 (with normals)
    # ***********************************
    #
    #
    
    print('\n--- QUESTION 4 ---\n')
    
    # Define parameters of recursive_RANSAC_with_normal
    nb_draws = 100
    threshold_in = 0.1
    threshold_angle = 0.2
    nb_planes = 5

    # Compute normals
    normals = compute_normal(points, radius=0.5)
    
    # Recursively find best plane by RANSAC
    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_RANSAC_with_normal(points, normals, nb_draws, threshold_in, threshold_angle, nb_planes)
    t1 = time.time()
    print('recursive RANSAC with normal done in {:.3f} seconds'.format(t1 - t0))
                
    # Save the best planes and remaining points
    write_ply('../best_planes_with_normal.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('../remaining_points_best_planes_with_normal.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    

    print('Done')
    