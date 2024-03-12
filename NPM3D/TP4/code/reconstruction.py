#
#
#      0===========================================================0
#      |              TP4 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 15/01/2024
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from skimage import measure

import trimesh


# Hoppe surface reconstruction
def compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel):
    # YOUR CODE    

    grid_nodes = np.meshgrid(
        np.arange(grid_resolution) * size_voxel + min_grid[0],
        np.arange(grid_resolution) * size_voxel + min_grid[1],
        np.arange(grid_resolution) * size_voxel + min_grid[2],
        indexing='ij'
    )
    grid_nodes = np.stack(grid_nodes, axis=-1).reshape(-1,3)

    tree = KDTree(points)
    closest_index = tree.query(grid_nodes, return_distance=False).squeeze()

    hoppe = np.sum(normals[closest_index] * (grid_nodes - points[closest_index]), axis=1)
    
    scalar_field[:,:,:] = hoppe.reshape(grid_resolution, grid_resolution, grid_resolution)


# IMLS surface reconstruction
def compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,knn):
    # YOUR CODE

    grid_nodes = np.meshgrid(
        np.arange(grid_resolution) * size_voxel + min_grid[0],
        np.arange(grid_resolution) * size_voxel + min_grid[1],
        np.arange(grid_resolution) * size_voxel + min_grid[2],
        indexing='ij'
    )
    grid_nodes = np.stack(grid_nodes, axis=-1).reshape(-1,3)

    tree = KDTree(points)
    closest_indices = tree.query(grid_nodes, k=knn, return_distance=False)

    h = 0.01

    diff = grid_nodes.reshape(-1,1,3) - points[closest_indices]
    theta = np.exp(-(np.linalg.norm(diff, axis=-1)**2 / (h**2)))
    imls = np.sum(np.sum(normals[closest_indices] * diff, axis=-1) * theta, axis=-1) / np.sum(theta, axis=-1)
    
    scalar_field[:,:,:] = imls.reshape(grid_resolution, grid_resolution, grid_resolution)



if __name__ == '__main__':

    t0 = time.time()
    
    # Path of the file
    file_path = '../data/bunny_normals.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

	# Compute the min and max of the data points
    min_grid = np.amin(points, axis=0)
    max_grid = np.amax(points, axis=0)
				
	# Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - 0.10*(max_grid-min_grid)
    max_grid = max_grid + 0.10*(max_grid-min_grid)

	# grid_resolution is the number of voxels in the grid in x, y, z axis
    grid_resolution = 128 # 16
    size_voxel = max([(max_grid[0]-min_grid[0])/(grid_resolution-1),(max_grid[1]-min_grid[1])/(grid_resolution-1),(max_grid[2]-min_grid[2])/(grid_resolution-1)])

	# Create a volume grid to compute the scalar field for surface reconstruction
    scalar_field = np.zeros((grid_resolution,grid_resolution,grid_resolution),dtype = np.float32)

	# Compute the scalar field in the grid
    # compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel)
    compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,30)

	# Compute the mesh from the scalar field based on marching cubes algorithm
    verts, faces, normals_tri, values_tri = measure.marching_cubes(scalar_field, level=0.0, spacing=(size_voxel,size_voxel,size_voxel))
    verts += min_grid
	
    # Export the mesh in ply using trimesh lib
    mesh = trimesh.Trimesh(vertices = verts, faces = faces)
    mesh.export(file_obj='../bunny_mesh_hoppe_16.ply', file_type='ply')
	
    print("Total time for surface reconstruction : ", time.time()-t0)
	


