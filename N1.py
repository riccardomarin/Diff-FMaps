import tensorflow_graphics.geometry.transformation as tfg_transformation
import tensorflow as tf
from open3d import *
import hdf5storage
import numpy as np
import sklearn
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from scipy import spatial

rand_seed = 1234
tf.random.set_seed(rand_seed) # Set random seed for deterministic experiments
tf.keras.backend.set_floatx('float32')


data = hdf5storage.loadmat('./clean_SMAL_dataset.mat') # Load dataset
PI  = hdf5storage.loadmat('./PI.mat')  # vertices subset
PI = PI['PI']
pix = np.squeeze(data['meshes_scaled'][1:8,PI-1,:])


def build_AE():   
    # AutoEncoder
    in_enc=tf.keras.layers.Input((pix.shape[1], 3), name='input')
    x = tf.keras.layers.Reshape((pix.shape[1], 3,1), name='res_input')(in_enc)
    x = tf.keras.layers.Conv2D(64, [1,3], strides=(1, 1), padding='valid', activation='relu', name='loc_1')(x)
    x = tf.keras.layers.BatchNormalization(name='loc_1_norm')(x)
    x = tf.keras.layers.Conv2D(64, [1,1], strides=(1, 1), padding='valid', activation='relu', name='loc_2')(x)
    point_feat = tf.keras.layers.BatchNormalization(name='loc_2_norm')(x)
    x = tf.keras.layers.Conv2D(64, [1,1], strides=(1, 1), padding='valid', activation='relu', name='glo_1')(point_feat)
    x = tf.keras.layers.BatchNormalization(name='glo_1_norm')(x)
    x = tf.keras.layers.Conv2D(128, [1,1], strides=(1, 1), padding='valid', activation='relu', name='glo_2')(x)
    x = tf.keras.layers.BatchNormalization(name='glo_2_norm')(x)
    x = tf.keras.layers.Conv2D(1024, [1,1], strides=(1, 1), padding='valid', activation='relu', name='glo_3')(x)
    x = tf.keras.layers.BatchNormalization(name='glo_3_norm')(x)
    glob_feat = tf.keras.layers.MaxPooling2D((pix.shape[1],1), name = 'glo_4')(x)


    global_feat_expand = tf.tile(glob_feat, [1, pix.shape[1], 1, 1])
    concat_feat = tf.concat([point_feat, global_feat_expand],3)
    x = tf.keras.layers.Conv2D(512, [1,1], strides=(1, 1), padding='valid', activation='relu', name='final_1')(concat_feat)
    x = tf.keras.layers.BatchNormalization(name='final_1_norm')(x)
    x = tf.keras.layers.Conv2D(256, [1,1], strides=(1, 1), padding='valid', activation='relu', name='final_2')(x)
    x = tf.keras.layers.BatchNormalization(name='final_2_norm')(x)
    x = tf.keras.layers.Conv2D(128, [1,1], strides=(1, 1), padding='valid', activation='relu', name='final_3')(x)
    x = tf.keras.layers.BatchNormalization(name='final_3_norm')(x)
    x = tf.keras.layers.Conv2D(128, [1,1], strides=(1, 1), padding='valid', activation='relu', name='final_4')(x)
    x = tf.keras.layers.BatchNormalization(name='final_4_norm')(x)
    x = tf.keras.layers.Conv2D(10, [1,1], strides=(1, 1), padding='valid', activation='linear', name='final_5')(x)
    x = tf.keras.layers.BatchNormalization(name='final_5_norm')(x)
    out = tf.keras.layers.Reshape((pix.shape[1], 10), name='res_out')(x)
    AE=tf.keras.Model(inputs=[in_enc],outputs=[out])
    return AE

AE = build_AE()

N1_optimizer = tf.keras.optimizers.Adam(1e-4)

BUFFER_SIZE = 60000 # Buffer for batch shuffle

b_size  = 8

n_batch = np.ceil(pix.shape[0]/32)
loss_all = np.zeros((100,))

perms = np.asarray([np.random.permutation(np.arange(0,pix.shape[1])) for x in np.arange(0,pix.shape[0])])
pix_perm = [x[perms[i]] for x,i in zip(pix,np.arange(0,4872))]
train_dataset_shapes = tf.data.Dataset.from_tensor_slices(pix_perm).shuffle(BUFFER_SIZE, seed=rand_seed).batch(np.int64(b_size)) # Training set batching

i=0
for epoch in range(0,100):
    print(epoch)
    loss_sum = 0;
    for meshes in train_dataset_shapes:
        with tf.GradientTape() as N1_tape, tf.GradientTape() as classer_tape:                      
            g_basis = AE((meshes), training=True)
            loss_b = tf.reduce_sum((tf.matmul(tf.transpose(g_basis,perm=[0, 2, 1]),g_basis) -  tf.tile(tf.reshape(tf.eye(10,10),(1,10,10)),(meshes.shape[0],1,1)))**2)

        gradients_of_generator = N1_tape.gradient([loss_b], AE.trainable_variables)
        temp = N1_optimizer.apply_gradients(zip(gradients_of_generator, AE.trainable_variables))
        loss_sum = loss_sum + loss_b.numpy()
    loss_all[i] = loss_sum / n_batch
    i=i+1;

plt.plot(loss_all)
plt.yscale('log')
plt.show()

data =np.squeeze( ((tf.matmul(tf.transpose(g_basis,perm=[0, 2, 1]),g_basis) -  tf.tile(tf.reshape(tf.eye(10,10),(1,10,10)),(meshes.shape[0],1,1))))[0])
plt.imshow(data)
plt.colorbar()

plt.show()
