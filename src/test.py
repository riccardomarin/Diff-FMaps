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
from plyfile import PlyData, PlyElement
import plot_utils
import scipy 
import getopt
parser = argparse.ArgumentParser()
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

GPU_INDEX = 0
BASE_LEARNING_RATE = 0.0001; DECAY_RATE = 0.7; DECAY_STEP = 8*250*200;
BATCH_SIZE = 1
NUM_POINT = 1000
N_BASIS = 20
LOG_FOUT = ''

sys.path.append('./src/models/')
MODEL = importlib.import_module('model')
TU = TrainUtils(float(DECAY_STEP), LOG_FOUT, BASE_LEARNING_RATE, DECAY_RATE, BATCH_SIZE)

def build_graph(MODEL_PATH):
    with tf.Graph().as_default() as g:
        with tf.device('/cpu:'+str(1)):
            
            is_training_pl = tf.constant(False)
            #batch = tf.Variable(0,name='batch')
            #bn_decay = TU.get_bn_decay(batch)
            with tf.variable_scope('basis'):
                pointclouds_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, N_BASIS)
                pred_basis = MODEL.get_basis_model(pointclouds_pl, is_training_pl, N_BASIS, bn_decay=None)
            with tf.variable_scope('transformation'):
                landmarksA = MODEL.compute_descriptors(pointclouds_pl,  is_training_pl, N_BASIS*2,bn_decay=None, icp=False)
            transformation_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='transformation')

            saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'pred_basis': pred_basis,
               'landmarksA': landmarksA
               }
        return sess, ops

def run_graph(sess, ops, shapes):
    """ ops: dict mapping from string to tf ops """

    feed_dict = {ops['pointclouds_pl']: shapes}
    pred_basis, descript = sess.run([ops['pred_basis'], ops['landmarksA']], feed_dict=feed_dict)

    return pred_basis, descript

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:n:e",["ifile=","ofile=","epoch="])
   except getopt.GetoptError:
      print ('test.py -i <inputfile> -e <experiment>')
      sys.exit(2)

   for opt, arg in opts:
      if opt == '-h':
         print ('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-n", "--network"):
         Network = arg
      elif opt in ("-p", "--epoch"):
         Epochs = str(arg)
   print('Input file is "', inputfile)
   return inputfile, Network, Epochs

if __name__ == "__main__":
    inputfile, Network, Epochs = main(sys.argv[1:])
    MODEL_PATH = './pretrained_models/' + Network + '/best_model_epoch_' + Epochs + '.ckpt'
    import os
    cwd = os.getcwd()
    print(cwd)
    print(MODEL_PATH)
    print('inputfile: ',inputfile); print('Network: ', Network); print('Epochs: ', Epochs)

    dataset =  scipy.io.loadmat('./data/'+inputfile)
    vertices = dataset['vertices']
    print(vertices.shape)
    sess, ops = build_graph(MODEL_PATH)
    pred_basis = []; pred_desc = []
    for i in range(vertices.shape[0]):
        basis, descr = run_graph(sess, ops, np.expand_dims(vertices[i],0))
        pred_basis.append(basis)
        pred_desc.append(descr)
    
    dataset['basis'] = np.squeeze(np.asarray(pred_basis))
    dataset['desc'] = np.squeeze(np.asarray(pred_desc))
    scipy.io.savemat('./evaluation/' + Network + '_' + Epochs + '_' + inputfile + '.mat', dataset)

    