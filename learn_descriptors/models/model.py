import tensorflow as tf
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/nn_distance'))

def placeholder_inputs(batch_size, num_point,n_basis):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    basis_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, n_basis))
    area_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_point))
    curv_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    perm_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, basis_pl, area_pl, curv_pl, perm_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """ ConvNet baseline, input is BxNx3 gray image """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1,3], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    points_feat1 = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    # MAX
    pc_feat1 = tf_util.max_pool2d(points_feat1, [num_point,1], padding='VALID', scope='maxpool1')
    # FC
    pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])
    pc_feat1 = tf_util.fully_connected(pc_feat1, 256, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    pc_feat1 = tf_util.fully_connected(pc_feat1, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)

    # CONCAT
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])
    points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])

    # CONV
    net = tf_util.conv2d(points_feat1_concat, 512, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv6')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv7')
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.conv2d(net, 30, [1,1], padding='VALID', stride=[1,1], # 30 is the number of descriptors
                         activation_fn=None, scope='conv8')
    net = tf.squeeze(net, [2])
    return net


def get_pred_C(basisA, basisB, descrA, descrB, areaA, areaB):
    """functional maps layer.
    Returns:
        Ct_est: estimated C (transposed), such that CA ~= B

    """
    basisA_trans =  tf.matmul(tf.transpose(basisA,[0, 2, 1]), areaA)
    basisB_trans =  tf.matmul(tf.transpose(basisB,[0, 2, 1]), areaB)
    descrA = tf.matmul(basisA_trans, descrA)
    descrB = tf.matmul(basisB_trans, descrB)
    # # Transpose input matrices
    descrA = tf.transpose(descrA, [0, 2, 1])
    descrB = tf.transpose(descrB, [0, 2, 1])
    #
    # # Solve C via least-squares
    Ct_est_AB = tf.matrix_solve_ls(descrA, descrB)
    C_est_AB = tf.transpose(Ct_est_AB, [0, 2, 1], name='C_est')
    return Ct_est_AB, C_est_AB


def distance_matrix(A, B):
    # nearest neighbour of A in B
    A = tf.expand_dims(A,-2)
    B = tf.expand_dims(B,-3)
    B = tf.tile(B, [1, 1000, 1, 1])
    A = tf.tile(A, [1, 1, 1000, 1])
    distances = tf.reduce_sum(tf.square(A - B), -1)
    return distances




def get_loss_sup(C_AB, basisA, basisB, T_AB, areaA, areaB):
    basisB_trans =  tf.matmul(tf.transpose(basisB,[0, 2, 1]), areaB)
    gt_C = tf.matmul(basisB_trans, tf.matmul(tf.transpose(T_AB, [0, 2, 1]), basisA))
    loss_sup = tf.reduce_mean(tf.square(C_AB - gt_C))
    return loss_sup, gt_C


def point2point_map_gt(perm):
    perm = tf.one_hot(perm, 1000, dtype=tf.float32)
    # T_AB = T_TB * T_TA^T
    T_AB_gt =tf.matmul(perm[:-1],tf.transpose(perm[1:], [0, 2, 1]))
    T_AB_gt = tf.transpose(T_AB_gt, [0, 2,1])
    _,indices_gt = tf.math.top_k(T_AB_gt)
    indices_gt = tf.squeeze(indices_gt)
    return T_AB_gt, indices_gt


def get_point2point_map(Ct_est_AB, evecsA, evecsB):
    EC =tf.matmul(evecsA, Ct_est_AB)
    distances = distance_matrix(EC, evecsB)
    dists,indices = tf.nn.top_k(-distances)
    indices = tf.squeeze(indices)
    return indices, distances

def get_loss_test(pc, basis, area, pred, perm):
    Ct_AB, C_AB= get_pred_C(basis[1:],basis[:-1], pred[1:], pred[:-1], area[1:] , area[:-1])
    T_AB_gt, indices_gt = point2point_map_gt(perm)
    loss_sup,gt_C = get_loss_sup(C_AB, basis[1:],basis[:-1],T_AB_gt,area[1:] , area[:-1])
    indices,nn_dists = get_point2point_map(Ct_AB,basis[1:],basis[:-1])
    return loss_sup,indices, indices_gt


# computes T using elemnt wise distance matrix
def get_T_AB(nn_dists):
    dists,indices = tf.nn.top_k(-nn_dists)
    # Create additional indices
    batch_size = nn_dists.shape[0]
    sequence_len = nn_dists.shape[1]
    sampled_size = 1
    i1, i2 = tf.meshgrid(tf.range(batch_size), tf.range(sequence_len), indexing="ij")
    i1 = tf.tile(i1[:, :, tf.newaxis], [1, 1, sampled_size])
    i2 = tf.tile(i2[:, :, tf.newaxis], [1, 1, sampled_size])
    # Create final indices
    idx = tf.stack([i1, i2, indices], axis=-1)
    T_AB = tf.scatter_nd(idx, dists,nn_dists.shape)
    T_AB = tf.divide(T_AB, tf.where( tf.equal(0.0, T_AB ), tf.ones_like( T_AB ), T_AB))
    return T_AB

def pseudo_euc_dist_loss(nn_dists, pc, T_AB_gt):
    distances = get_T_AB(nn_dists)
    features_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.matmul(distances, pc[:-1]) - tf.matmul(T_AB_gt, pc[:-1])),-1))
    return features_loss

def get_loss(pc, basis, area,curv, pred, perm, laplacian, part_labels):
    Ct_AB, C_AB= get_pred_C(basis[1:],basis[:-1], pred[1:], pred[:-1], area[1:] , area[:-1])
    Ct_BA, C_BA= get_pred_C(basis[:-1],basis[1:],  pred[:-1],pred[1:],area[:-1], area[1:])
    indices,nn_dists = get_point2point_map(Ct_AB,basis[1:],basis[:-1])
    T_AB_gt, indices_gt = point2point_map_gt(perm)
    euc_dist = pseudo_euc_dist_loss(nn_dists, pc, T_AB_gt)
    loss_sup,gt_C = get_loss_sup(C_AB, basis[1:],basis[:-1],T_AB_gt,area[1:] , area[:-1])
    return loss_sup, euc_dist
