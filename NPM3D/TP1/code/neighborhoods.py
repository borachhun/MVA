#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):

    # YOUR CODE

    # neighborhoods = list of numpy arrays where each
    # numpy array contains neighbors of one query
    neighborhoods = []

    for i in range(queries.shape[0]):

        # Compute square distance from a query to the supports 
        sqrt_distance_from_query = ((supports - queries[i])**2).sum(axis=1)

        # (distance from query to support)^2 <= radius^2
        neighbors = np.where(sqrt_distance_from_query <= radius**2)[0]

        neighborhoods.append(supports[neighbors, :])

    return neighborhoods


def brute_force_KNN(queries, supports, k):

    # YOUR CODE

    # neighborhoods = list of numpy arrays where each
    # numpy array contains neighbors of one query
    neighborhoods = []

    for i in range(queries.shape[0]):

        # Compute square distance from a query to the supports 
        sqrt_distance_from_query = ((supports - queries[i])**2).sum(axis=1)

        # Take k points with smallest distance from query
        neighbors = np.argsort(sqrt_distance_from_query)[:k]

        neighborhoods.append(supports[neighbors, :])

    return neighborhoods





# ------------------------------------------------------------------------------------------
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

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if True:

        # Define the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))

 



    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if True:

        # Define the search parameters
        num_queries = 1000

        # YOUR CODE
        leaf_sizes = [1, 2, 5, 10, 50, 100, 200]
        time_list = []
        for ls in leaf_sizes:
            tree = KDTree(points, leaf_size=ls)

            random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
            queries = points[random_indices, :]

            start_time = time.time()
            tree.query_radius(queries, r=0.2)
            end_time = time.time()
            time_list.append(end_time - start_time)
        
        plt.plot(leaf_sizes, time_list)
        plt.title('Time for 1000 queries')
        plt.xlabel('Leaf size')
        plt.ylabel('Time in second')
        plt.show()



        tree = KDTree(points, leaf_size=50)
        radius_list = [0.1, 0.2, 0.5, 0.75, 1, 1.2, 1.5]
        time_list = []
        for radius in radius_list:
            random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
            queries = points[random_indices, :]

            start_time = time.time()
            tree.query_radius(queries, r=radius)
            end_time = time.time()
            time_list.append(end_time - start_time)
        
        plt.plot(radius_list, time_list)
        plt.title('Time for 1000 queries')
        plt.xlabel('Radius')
        plt.ylabel('Time in second')
        plt.show()

        print('Time to search 20cm neighborhoods for all points in the cloud:',
              time_list[1] / num_queries * points.shape[0], 'seconds')
