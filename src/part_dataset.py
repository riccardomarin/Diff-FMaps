'''
    Dataset for shapenet part segmentaion.
'''
import os
import os.path
import json
import numpy as np
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from scipy.io import loadmat
from plyfile import PlyData
import trimesh

def rotate_point_cloud(batch_data_shuffled, batch_data ):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotated_data_shuffled = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        shape_pc_shuffled = batch_data_shuffled[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data_shuffled[k, ...] = np.dot(shape_pc_shuffled.reshape((-1, 3)), rotation_matrix)
    return rotated_data_shuffled, rotated_data


class PartDataset():
    def __init__(self, root, split='train', limit=2000):
        """
        root: data folder
        split: train or test
        limit: load shapes from 1 to LIMIT
        """

        print('loading...')

        if split =="12ktrain":
            print('loading train shapes')
            self.data = np.load(os.path.join(root, "12k_shapes_train.npy"))
            EDGES_PATH = os.path.join(root,"12ktemplate.ply")
        elif split =="12ktest":
            print('loading test shapes')
            self.data = np.load(os.path.join(root, "12k_shapes_test.npy"))
            EDGES_PATH = os.path.join(root,"12ktemplate.ply")
        else:
            print('unknown split')
        self.NUM_POINTS = self.data[0].shape[0]

        plydata = PlyData.read(EDGES_PATH)
        FACES = plydata['face']
        FACES = np.array([FACES[i][0] for i in range(FACES.count)])
        mesh = trimesh.Trimesh(self.data[0],FACES, process = False)
        self.f = FACES
        print(split + ' data loaded')

    def __getitem__(self, index):
        point_set = self.data[index]
        choice = range(self.NUM_POINTS)
        point_set_shuffled = point_set[choice, :]
        return point_set_shuffled, point_set, choice


    def __len__(self):
        return len(self.data)
