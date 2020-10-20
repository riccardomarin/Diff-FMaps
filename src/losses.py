import tensorflow as tf
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/nn_distance'))

########### INVARIANCE LOSSES ###########
def optimal_transformation(X, Y):
    # returns X to which we apply the optimal transformation between  X and Y
    n_pc_points = X.shape[1]
    n_dim = X.shape[2]
    n_couples = X.shape[0]
    mu_x = tf.reduce_mean(X, axis = 1)
    mu_y =  tf.reduce_mean(Y, axis = 1)
    concat_mu_x = tf.tile(tf.expand_dims(mu_x,1), [1, n_pc_points, 1])
    concat_mu_y = tf.tile(tf.expand_dims(mu_y,1), [1, n_pc_points, 1])
    centered_y = tf.expand_dims(Y - concat_mu_y, 2)
    centered_x = tf.expand_dims(X - concat_mu_x, 2)

    # transpose y
    centered_y = tf.einsum('ijkl->ijlk', centered_y)
    mult_xy = tf.einsum('abij,abjk->abik', centered_y, centered_x)
    # sum
    C = tf.einsum('abij->aij', mult_xy)
    s, u,v = tf.svd(C)
    v = tf.einsum("aij->aji", v)
    # prevent reflections
    diag_mult = tf.matrix_diag(tf.concat([tf.ones([n_couples, n_dim-1]), tf.linalg.det(tf.matmul(u,v))[:,tf.newaxis]],axis = 1))
    R_opt = tf.matmul(tf.matmul(u,diag_mult),v)
    t_opt = mu_y - tf.einsum("aki,ai->ak", R_opt, mu_x)
    concat_R_opt = tf.tile(tf.expand_dims(R_opt,1), [1, n_pc_points, 1, 1])
    concat_t_opt = tf.tile(tf.expand_dims(t_opt,1), [1, n_pc_points, 1])
    opt_X =  tf.einsum("abki,abi->abk", concat_R_opt, X) + concat_t_opt
    return opt_X

def optimal_linear_transformation(X, Y):
    # returns X to which we apply the optimal transformation between  X and Y
    pseudo_inv_X = tf.linalg.pinv(X)
    C_transp = tf.matmul(pseudo_inv_X, Y)
    opt_X = tf.matmul(X, C_transp)
    return opt_X, C_transp

def distance_matrix(A, B):
    # nearest neighbour of A in B
    sizeA = A.shape[1]
    sizeB = B.shape[1]
    A = tf.expand_dims(A,-2)
    B = tf.expand_dims(B,-3)
    B = tf.tile(B, [1, sizeA, 1, 1])
    A = tf.tile(A, [1, 1, sizeB, 1])
    distances = tf.reduce_sum(tf.square(A - B), -1)
    return distances

def softmax_dist(basisA, basisB):
    distances = distance_matrix(basisA, basisB)
    smax = tf.math.softmax(-distances, axis =1)
    return smax

def euc_dist_err(pcB, smax):
    match = tf.matmul(smax,pcB)
    return tf.reduce_sum(tf.square(pcB - match))

def get_basis_loss(pcA, pcB, basisA, basisB):
    basisA_transf, C = optimal_linear_transformation(basisA, basisB)
    smax = softmax_dist(basisA_transf, basisB)
    euc_dist = euc_dist_err(pcB, smax)
    return euc_dist


def get_descriptors_loss(pred_basisA, descriptorsA,pred_basisB, descriptorsB):
    spectral_A = tf.matmul(tf.linalg.pinv(pred_basisA), descriptorsA)
    spectral_B = tf.matmul(tf.linalg.pinv(pred_basisB), descriptorsB)
    spectral_At = tf.transpose(spectral_A, [0, 2, 1])
    spectral_Bt = tf.transpose(spectral_B, [0, 2, 1])
    tran = tf.matmul(tf.linalg.pinv(spectral_At),spectral_Bt)
    basisA_transf, C = optimal_linear_transformation(pred_basisA, pred_basisB)
    loss = tf.reduce_mean(tf.square(C - tran))
    return loss



