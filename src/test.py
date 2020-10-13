import argparse
from args_to_flag import args_to_flag
import os
import sys
# Paths setup
try:   # To handle both bash and file execution
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except:
    BASE_DIR = os.getcwd()

ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'losses'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))
from train_utils import TrainUtils
from datetime import datetime
import math
import h5py
import numpy as np
import tensorflow as tf
import importlib
import part_dataset
import  losses #This is where we organize our losses
from plyfile import PlyData, PlyElement
import plot_utils
import scipy
import sklearn.neighbors
import scipy.io as sio
from collections import defaultdict
N_BASIS = 20
seed = 1            #For deterministic experiments, used for numpy and TF
np.random.seed(seed)

# Args parsing and setup
parser = argparse.ArgumentParser()

PATH_MODEL, _, BATCH_SIZE, NUM_POINT, _, BASE_LEARNING_RATE, GPU_INDEX, _, _, DECAY_STEP, DECAY_RATE, LOG_FOUT, _, TRAINED_MODEL_PATH = args_to_flag(parser, train=False)
DATA_ROOT = "./data/"
EDGES_PATH = "./data/12ktemplate.ply"

EPOCH_CNT = 0
MODEL_FILE = os.path.join(BASE_DIR, PATH_MODEL + '.py')


TEST_DATASET  = part_dataset.PartDataset(root=DATA_ROOT, split='12ktest')

BATCH_SIZE = 16

MODEL = importlib.import_module(PATH_MODEL) # import network module

TU = TrainUtils(float(DECAY_STEP), LOG_FOUT, BASE_LEARNING_RATE, DECAY_RATE, BATCH_SIZE)

plydata = PlyData.read(EDGES_PATH)
FACES = plydata['face']
FACES = np.array([FACES[i][0] for i in range(FACES.count)])
dd = sio.loadmat('./data/mean_dist.mat')
def build_graph():
    with tf.Graph().as_default() as g:
        with tf.device('/gpu:'+str(GPU_INDEX)):
            tf.set_random_seed(seed)

            batch = tf.Variable(0)
            bn_decay = TU.get_bn_decay(batch)
            mask = tf.constant(dd['mean_dist'],tf.float32)

            # Input placeholders
            print("--- Set PlaceHolders")
            pointclouds_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, N_BASIS)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            print("--- Get model and loss")
            # Basis model
            with tf.variable_scope('basis'):
                pointcloudsA_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, N_BASIS)
                pointcloudsB_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, N_BASIS)
                is_training_pl = tf.constant(False)
                pred_basisA, latentA = MODEL.get_basis_model(pointcloudsA_pl, is_training_pl, N_BASIS, bn_decay=None)
                pred_basisB, latentB = MODEL.get_basis_model(pointcloudsB_pl, is_training_pl, N_BASIS, bn_decay=None)
            with tf.variable_scope("transformation"):
                landmarksA = MODEL.compute_descriptors(pointcloudsA_pl,  is_training_pl, N_BASIS*2,bn_decay=None, icp=False)
                landmarksB = MODEL.compute_descriptors(pointcloudsB_pl,  is_training_pl, N_BASIS*2,bn_decay=None, icp=False)

                # test
                spectral_A = tf.matmul(tf.linalg.pinv(pred_basisA), landmarksA)
                spectral_B = tf.matmul(tf.linalg.pinv(pred_basisB), landmarksB)
                spectral_At = tf.transpose(spectral_A, [0, 2, 1])
                spectral_Bt = tf.transpose(spectral_B, [0, 2, 1])
                tran = tf.matmul(tf.linalg.pinv(spectral_At),spectral_Bt)
                basisA_tran = tf.matmul(pred_basisA, tran)
                X, C_opt = losses.optimal_linear_transformation(pred_basisA, pred_basisB)
                direct_loss_old = tf.reduce_mean(tf.square(pred_basisA - pred_basisB))
                direct_loss = tf.reduce_mean(tf.square(basisA_tran - pred_basisB))
            transformation_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='transformation')
            saver= tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Init variables
        init = tf.global_variables_initializer()
        sess = tf.Session(config=config)
        sess.run(init)
        saver.restore(sess,TRAINED_MODEL_PATH)

        ops = {'pointcloudsA_pl': pointcloudsA_pl,
                'pointcloudsB_pl': pointcloudsB_pl,
               'pred_basisA': pred_basisA,
               'pred_basisB': pred_basisB,
               'transform': tran,
               'landmarksA': landmarksA,
               'basisA_tran': basisA_tran,
               'C_opt': C_opt,
               'landmarksB': landmarksB,
               'direct_loss': direct_loss,
               'direct_loss_old': direct_loss_old,
               }
        return sess, ops

def matching_pairs(test_idx, src_idx, tar_idx):
    src,  batch_data, permutation_data = TU.get_batch(TEST_DATASET, src_idx, 0, len(src_idx))
    tar,  batch_data, permutation_data = TU.get_batch(TEST_DATASET, tar_idx, 0, len(tar_idx))

    # Reorganize vertex order (just to check network is invariant to order)
    choice_src = [np.array(range(1000)) for x in range(len(src_idx))]

    src = np.asarray([src[i][choice_src[i],:] for i in range(0,len(choice_src))])

    choice_tar = [np.array(range(1000)) for x in range(len(tar_idx))]
    tar = np.asarray([tar[i][choice_tar[i],:] for i in range(0,len(choice_tar))])

    # compute GT correspondence between reorganized pointclouds
    corr  = np.arange(0,src.shape[1])
    gt = np.zeros((src.shape[0],src.shape[1]),np.int32) # build an operator: SRC_p -> TAR_p
    for i in range(0,src.shape[0]):
        perm1 = choice_src[i]
        perm2 = choice_tar[i]
        perm1_matrix = scipy.sparse.csc_matrix((np.ones((1000,)), (np.arange(0,1000),perm1)),shape = (1000,1000)) # SRC -> SRC_p
        perm2_matrix = scipy.sparse.csc_matrix((np.ones((1000,)), (np.arange(0,1000),perm2)),shape = (1000,1000)) # TAR -> TAR_p
        corr_matrix  = scipy.sparse.csc_matrix((np.ones((1000,)), (np.arange(0,1000),corr )),shape = (1000,1000)) # SRC -> TAR
        perm1_inv_matrix = perm1_matrix.T                                                                         # SRC_p -> SRC
        corr_original = np.matmul(perm2_matrix.todense(), np.matmul(corr_matrix.todense(), perm1_inv_matrix.todense())) # SRC_p -> SRC -> TAR -> TAR_p
        gt[i] = np.where(corr_original)[1]
        assert np.count_nonzero(choice_src[i][gt[i]] - choice_tar[i]) == 0
    return src, tar, choice_src, choice_tar, gt

def run_graph(sess, ops, src, tar):
    """ ops: dict mapping from string to tf ops """

    feed_dict = {ops['pointcloudsA_pl']: src,
                ops['pointcloudsB_pl']: tar}
    direct_loss_old,landmarksA,landmarksB,direct_loss, pred_basis_src,pred_basis_tar,basisA_tran,transform,C_opt= sess.run(
        [ops['direct_loss_old'],ops['landmarksA'],ops['landmarksB'],ops['direct_loss'],ops['pred_basisA'],ops['pred_basisB'],ops['basisA_tran'],ops['transform'],ops['C_opt']], feed_dict=feed_dict)

    return direct_loss_old,landmarksA,landmarksB,direct_loss,pred_basis_src, basisA_tran, pred_basis_tar, transform,C_opt

def matching(pred_basis_src, pred_basis_tar):
    matches = np.zeros((pred_basis_src.shape[0],pred_basis_src.shape[1]))
    for i in range(pred_basis_src.shape[0]):
        source = pred_basis_src[i,:,:]
        target = pred_basis_tar[i,:,:]
        tree = sklearn.neighbors.KDTree(target)
        matches[i] = np.squeeze(tree.query(source)[1])
    return matches

def test():
    sess, ops = build_graph()

    # Generate matching pairs
    test_idxs = np.arange(0, 33)#np.arange(0, len(TEST_DATASET))
    src_idx = test_idxs[1:]   # SRC indices
    tar_idx = test_idxs[:-1]  # TAR indices

    predicted_mappings = defaultdict(list)
    predicted_mappings['f'] = FACES

    for batch_idx in range(len(test_idxs)//(BATCH_SIZE)):
        src, tar, choice_src, choice_tar, gt = matching_pairs(test_idxs, test_idxs[batch_idx*BATCH_SIZE + 1:(batch_idx+1)*BATCH_SIZE + 1], test_idxs[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE])

        # Generate basis for matching
        direct_loss_old,landmarksA,landmarksB, direct_loss_val, pred_basis_src, basisA_tran, pred_basis_tar, transform,C_opt = run_graph(sess, ops, src, tar)
        print('direct_loss old : ', direct_loss_old)
        print("direct_loss : ", direct_loss_val)

        print('diff learned',np.mean(np.square(C_opt - transform)))
        matches = matching(pred_basis_src, pred_basis_tar)
        matchesT = matching(basisA_tran, pred_basis_tar)
        predicted_mappings['Src'].append(src)
        predicted_mappings['Tar'].append(tar)
        predicted_mappings['SrcBasis'].append(pred_basis_src)
        predicted_mappings['TarBasis'].append(pred_basis_tar)
        predicted_mappings['SrcBasisT'].append(basisA_tran)
        predicted_mappings['SrcChoice'].append(choice_src)
        predicted_mappings['TarChoice'].append(choice_tar)
        predicted_mappings['matches'].append(matches)
        predicted_mappings['matchesT'].append(matchesT)
        predicted_mappings['gt'].append(gt)
        predicted_mappings['transform'].append(transform)
    for k in predicted_mappings.keys():
        if k!='f':
            predicted_mappings[k] = np.concatenate(predicted_mappings[k])
    scipy.io.savemat('results/output_12k.mat',predicted_mappings)

if __name__ == "__main__":
    test()
