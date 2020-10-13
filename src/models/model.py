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
    return pointclouds_pl
def placeholder_basis(batch_size, num_point,n_basis):
    basis_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, n_basis))
    return basis_pl


def compute_descriptors(basisA, is_training, n_basis,bn_decay=None, icp=False):
    batch_size = basisA.get_shape()[0].value
    num_point = basisA.get_shape()[1].value
    input_image = tf.expand_dims(basisA, -1)
    net1 = tf_util.conv2d(input_image, 64, [1,basisA.get_shape()[2].value], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv1b', bn_decay=bn_decay)
    net1 = tf_util.conv2d(net1, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv2b', bn_decay=bn_decay)
    net1 = tf_util.conv2d(net1, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv3b', bn_decay=bn_decay)
    net1 = tf_util.conv2d(net1, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv4b', bn_decay=bn_decay)
    points_feat1 = tf_util.conv2d(net1, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv5b', bn_decay=bn_decay)
    pc_feat1 = tf_util.max_pool2d(points_feat1, [num_point,1], padding='VALID', scope='maxpool1b')

    # MAX

    # FC
    pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])
    pc_feat1 = tf_util.fully_connected(pc_feat1, 256, bn=True, is_training=is_training, scope='fc1b', bn_decay=bn_decay)
    pc_feat1 = tf_util.fully_connected(pc_feat1, 256, bn=True, is_training=is_training, scope='fc2b', bn_decay=bn_decay)

    # CONCAT
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])
    points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])
    # CONV
    net1 = tf_util.conv2d(points_feat1_concat, 512, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv6b')
    net1 = tf_util.conv2d(net1, 256, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv7b')
    #net1 = tf_util.dropout(net1, keep_prob=0.7, is_training=is_training, scope='dp1b')
    net1 = tf_util.conv2d(net1, n_basis, [1,1], padding='VALID', stride=[1,1], # 30 is the number of descriptors
                         activation_fn=None, scope='conv8b')
    landmarks = tf.squeeze(net1, [2])
    #landmarks = tf.nn.softmax(landmarks, axis = 1)
    #landmarks_index = tf.argmax(landmarks, axis = 1)
    #landmarks = tf.gather(basisA, landmarks_index)

    return landmarks

def get_basis_model(point_cloud, is_training, n_basis, bn_decay=None):
    #BASIS
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)
    net1 = tf_util.conv2d(input_image , 64, [1,point_cloud.get_shape()[-1].value], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv1b', bn_decay=bn_decay)
    net1 = tf_util.conv2d(net1, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv2b', bn_decay=bn_decay)
    net1 = tf_util.conv2d(net1, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv3b', bn_decay=bn_decay)
    net1 = tf_util.conv2d(net1, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv4b', bn_decay=bn_decay)
    points_feat1 = tf_util.conv2d(net1, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv5b', bn_decay=bn_decay)
    # MAX
    pc_feat1 = tf_util.max_pool2d(points_feat1, [num_point,1], padding='VALID', scope='maxpool1b')
    # FC
    pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])
    pc_feat1 = tf_util.fully_connected(pc_feat1, 256, bn=True, is_training=is_training, scope='fc1b', bn_decay=bn_decay)
    pc_feat1 = tf_util.fully_connected(pc_feat1, 128, bn=True, is_training=is_training, scope='fc2b', bn_decay=bn_decay)

    # CONCAT
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])
    points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])

    # CONV
    net1 = tf_util.conv2d(points_feat1_concat, 512, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv6b')
    net1 = tf_util.conv2d(net1, 256, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv7b')
    net1 = tf_util.dropout(net1, keep_prob=0.7, is_training=is_training, scope='dp1b')
    net1 = tf_util.conv2d(net1, n_basis, [1,1], padding='VALID', stride=[1,1], # 30 is the number of descriptors
                         activation_fn=None, scope='conv8b')
    basis = tf.squeeze(net1, [2])
    return basis

#
