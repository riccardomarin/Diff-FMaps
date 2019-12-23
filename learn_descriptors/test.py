import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import open3d
import matplotlib
import matplotlib.cm as cm

from plyfile import PlyData, PlyElement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))
import part_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
#parser.add_argument('--model', default='model_segm', help='Model name [default: model]')
parser.add_argument('--model', default='model_clean', help='Model name [default: model]')

parser.add_argument('--log_dir', default='log_corres', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1000, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 32]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
#N_BASIS = 40
N_BASIS = 10

# Shapenet official train/test split
DATA_ROOT = "/media/marie-julie/DATAPART2/differentiable_FM/surreal"
TEST_DATASET = part_dataset.PartDataset(root=DATA_ROOT, npoints=NUM_POINT,  split='test',normalize_area = False, normalize=True,n_basis = N_BASIS)

EDGES_PATH = "/media/marie-julie/DATAPART2/datasets/DFAUST_SURREAL/template.ply"
plydata = PlyData.read(EDGES_PATH)
FACES = plydata['face']
FACES = np.array([FACES[i][0] for i in range(FACES.count)])

def init_pc(v,f,colors, offset=[0,0,0]):
    pc = open3d.TriangleMesh()
    pc.vertices = open3d.Vector3dVector(v + offset)
    pc.triangles = open3d.Vector3iVector(f)
    pc.vertex_colors = open3d.Vector3dVector(colors[:,0:3])
    return pc


def plot_f(gt_v,v,f,func1, func2):
    print('plotting...')
    cmap = cm.get_cmap('Spectral')
    func1 = (func1 - np.min(func1)) / (np.max(func1) - np.min(func1))
    colors = cmap(func1)
    pc1 = init_pc(v,f,colors, offset = [1,0,0])

    func2 = (func2 - np.min(func2)) / (np.max(func2) - np.min(func2))
    colors = cmap(func2)
    pc2 = init_pc(gt_v, f, colors)
    open3d.draw_geometries([pc1, pc2])


def plot_f2(gt_v,v,f,func1, func2):
    print('plotting...')
    cmap = cm.get_cmap('Spectral')
    min_val = np.min([np.min(func1,0),np.min(func2,0)], 0)
    max_val = np.min([np.max(func1,0),np.max(func2,0)], 0)
    func1 = (func1 - min_val)/np.tile((max_val-min_val),(np.size(func1,0),1))
    func2 = (func2 - min_val)/np.tile((max_val-min_val),(np.size(func2,0),1))
    colors = np.concatenate([func1, np.ones((func1.shape[0], 1))], 1)
    pc1 = init_pc(v,f,colors, offset = [1,0,0])

    colors = np.concatenate([func2, np.ones((func1.shape[0], 1))], 1)
    pc2 = init_pc(gt_v, f, colors)
    open3d.draw_geometries([pc1, pc2])

def test():
    with tf.Graph().as_default() as g:
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl,basis_pl, area_pl,_, perm_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, N_BASIS)
            is_training_pl = tf.constant(False)
            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            # Get model and loss
            with tf.variable_scope("reconstruction"):
                pred = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=None)

                #pred  = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 6))

                loss,indx, indx_gt = MODEL.get_loss_test(pointclouds_pl, basis_pl, area_pl, pred, perm_pl)
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        # Init variables
        init = tf.global_variables_initializer()

        sess.run(init)
        #MODEL_PATH = 'log_corres/best_model_epoch_1279.ckpt' # BASIS: 40, descriptors: 30
        MODEL_PATH = 'log_coorespondances_supervised/best_model_epoch_032.ckpt' # BASIS: 10, descriptors: 30

        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'basis_pl' : basis_pl,
               'area_pl' : area_pl,
               'loss': loss,
               'perm_pl': perm_pl,
               'indx' :indx ,
               'indx_gt' :indx_gt,
               #'parts': pred
               }


        eval_one_epoch(sess, ops)


def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_data_shuffled = np.zeros((bsize, NUM_POINT, 3))
    basis_data_shuffled = np.zeros((bsize, NUM_POINT, N_BASIS))
    area_data_shuffled = np.zeros((bsize, NUM_POINT,NUM_POINT))
    permutation_data = np.zeros((bsize, NUM_POINT))
    parts = np.zeros((bsize, NUM_POINT, 6))

    for i in range(bsize):
        point_set_shuffled,basis_shuffled, area_shuffled,_, point_set,permutation = dataset[idxs[i+start_idx]]
        batch_data[i,...] = point_set
        batch_data_shuffled[i,...] = point_set_shuffled
        basis_data_shuffled[i,...] = basis_shuffled
        area_data_shuffled[i,...] = area_shuffled
        permutation_data[i,...] = permutation
        parts[i,...] = TEST_DATASET.part_labels[permutation]
    return batch_data_shuffled,basis_data_shuffled, area_data_shuffled,  batch_data, permutation_data, parts

def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    test_idxs = np.arange(0, len(TEST_DATASET))
    np.random.shuffle(test_idxs)
    num_batches = len(TEST_DATASET)/BATCH_SIZE

    loss_sum = 0
    for batch_idx in range(10):#range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data_shuffled,basis_data_shuffled, area_data_shuffled,  batch_data, permutation_data, parts_labels  = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)
        batch_data_shuffled, batch_data = part_dataset.rotate_point_cloud(batch_data_shuffled, batch_data )
        feed_dict = {ops['pointclouds_pl']: batch_data_shuffled,
                     ops['labels_pl']: batch_data,
                     ops['basis_pl'] : basis_data_shuffled,
                     ops['area_pl']: area_data_shuffled,
                     ops['perm_pl'] : permutation_data}
                     #ops['parts']: parts_labels}
        loss_val, indx, indx_gt, perm = sess.run([ops['loss'], ops['indx'], ops['indx_gt'], ops['perm_pl']], feed_dict=feed_dict)
        for i in range(len(indx)):
            # source_x_func = batch_data_shuffled[i,:,0]
            # target_pred_func = source_x_func[indx[i]][np.argsort(perm[i+1])]
            # target_gt_func = source_x_func[indx_gt[i]][np.argsort(perm[i+1])]
            # v = batch_data_shuffled[i+1][np.argsort(perm[i+1])]
            # source_v = batch_data_shuffled[i][np.argsort(perm[i])]
            # plot_f(source_v, v, FACES, target_pred_func, target_gt_func)
            source_x_func = batch_data_shuffled[i]
            target_pred_func = source_x_func[indx[i]][np.argsort(perm[i+1])]
            target_gt_func = source_x_func[indx_gt[i]][np.argsort(perm[i+1])]
            b = basis_data_shuffled[i][indx_gt[i]][np.argsort(perm[i+1])]
            a = area_data_shuffled[i][indx_gt[i]][np.argsort(perm[i+1])]
            #target_gt_func = np.matmul(b, np.matmul(np.transpose(b), np.matmul(a, target_gt_func)))

            v = batch_data_shuffled[i+1][np.argsort(perm[i+1])]
            source_v = batch_data_shuffled[i][np.argsort(perm[i])]
            plot_f2(source_v, v, FACES, target_pred_func, target_gt_func)

        loss_sum += loss_val
        print(loss_val)
    return loss_sum/float(num_batches)


if __name__ == "__main__":
    test()
