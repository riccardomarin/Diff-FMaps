import argparse
from args_to_flag import args_to_flag
from train_utils import TrainUtils
from datetime import datetime
import os
import sys
import math
import h5py
import numpy as np
import tensorflow as tf
import importlib
import part_dataset
import losses
import scipy.io as sio
####### Parameters required

N_BASIS = 20

seed = 1                   
np.random.seed(seed)
# Args parsing and setup
parser = argparse.ArgumentParser()
PATH_MODEL, LOG_DIR, BATCH_SIZE, NUM_POINT, MAX_EPOCH, BASE_LEARNING_RATE, GPU_INDEX, MOMENTUM, OPTIMIZER, DECAY_STEP, DECAY_RATE, LOG_FOUT, NO_ROTATION,_ = args_to_flag(parser)

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
DATA_ROOT = "./data/"

MODEL_FILE = os.path.join(BASE_DIR, PATH_MODEL + '.py')

# Data loading
TRAIN_DATASET = part_dataset.PartDataset(root=DATA_ROOT, split='12ktrain', limit=10000)
TEST_DATASET  = part_dataset.PartDataset(root=DATA_ROOT, split='12ktest' , limit= 2000)

MODEL = importlib.import_module(PATH_MODEL) # import network module
TU = TrainUtils(float(DECAY_STEP), LOG_FOUT, BASE_LEARNING_RATE, DECAY_RATE, BATCH_SIZE)

EPOCH_CNT = 0

def build_graph():
    with tf.Graph().as_default() as g:
        with tf.device('/gpu:'+str(GPU_INDEX)):
            tf.set_random_seed(seed)

            batch = tf.Variable(0)
            bn_decay = TU.get_bn_decay(batch)

            # Input placeholders
            print("--- Set PlaceHolders")
            pointclouds_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, N_BASIS)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            print("--- Get model and loss")
            with tf.variable_scope('basis'):
                # Basis model
                pred_basis = MODEL.get_basis_model(pointclouds_pl, is_training_pl, N_BASIS, bn_decay=bn_decay)
            # Correspondence couples
            basisA = pred_basis[1:]; pcA = pointclouds_pl[1:]
            basisB = pred_basis[:-1]; pcB = pointclouds_pl[:-1]
            # Check if we optimize also Transformation Matrix
            euc_dist_loss = losses.get_basis_loss(pcA, pcB, basisA, basisB)
            total_loss = euc_dist_loss

            print("--- Get training operator")
            # Get training operator
            learning_rate = TU.get_learning_rate(batch)

            # global step for adding volume loss
            global_step = tf.Variable(-0.20, name='global_step', trainable=False, dtype=tf.float32)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            init_op = optimizer.minimize(total_loss, global_step=batch)

            # add summary to observe variables
            tf.summary.scalar('bn_decay', bn_decay)
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('euc dist', euc_dist_loss)
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # Operations dictionary
        ops = {'pointclouds_pl': pointclouds_pl,
               'is_training_pl': is_training_pl,
               'init_op': init_op,
               'total_loss':total_loss,
               'merged': merged,
               'step': batch
               }

        return sess, ops, train_writer, test_writer, saver

def train():
    sess, ops, train_writer, test_writer, saver = build_graph()
    best_loss = 1e20
    for epoch in range(MAX_EPOCH):
        TU.log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()

        # Train
        train_one_epoch(sess, ops, train_writer,ops['init_op'], epoch)
        # Validation
        epoch_loss = eval_one_epoch(sess, ops, test_writer)

        # Best Model Checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
            TU.log_string("Model saved in file: %s" % save_path)

        # Temporal Model Checkpoint
        if epoch % 10 == 0:
            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
            TU.log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer, train_op, epoch):
    is_training = True
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)//BATCH_SIZE
    TU.log_string(str(datetime.now()))

    loss_sum = 0
    for batch_idx in np.arange(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data_shuffled, batch_data, permutation_data = TU.get_batch(TRAIN_DATASET, train_idxs, np.int32(start_idx), np.int32(end_idx))
        # Augment batched point clouds by rotation
        if NO_ROTATION:
            aug_data = batch_data_shuffled
            aug_data_unshuffled = batch_data
        else:
            aug_data, aug_data_unshuffled = part_dataset.rotate_point_cloud(batch_data_shuffled, batch_data )

        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['is_training_pl']: is_training
                     }

        summary, step, _,  total_loss_val= sess.run([ops['merged'], ops['step'], train_op,  ops['total_loss']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        loss_sum+=total_loss_val

        if (batch_idx+1)%10 == 0:
            TU.log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))

def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = np.int32(len(TEST_DATASET)/BATCH_SIZE)
    print(num_batches)
    TU.log_string(str(datetime.now()))
    TU.log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    loss_sum = 0
    for batch_idx in np.arange(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data_shuffled, batch_data, permutation_data  = TU.get_batch(TEST_DATASET, test_idxs, np.int32(start_idx), np.int32(end_idx))

        feed_dict = {ops['pointclouds_pl']: batch_data_shuffled,
                     ops['is_training_pl']: is_training}

        summary, step, loss_val = sess.run([ops['merged'], ops['step'],ops['total_loss']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        test_writer.flush()
        loss_sum += loss_val
    TU.log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))

    EPOCH_CNT += 1

    return loss_sum/float(num_batches)


if __name__ == "__main__":
    TU.log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
