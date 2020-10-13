
import tensorflow as tf
import numpy as np

class TrainUtils():
    def __init__(self, DECAY_STEP, LOG_FOUT,BASE_LEARNING_RATE,DECAY_RATE,BATCH_SIZE):
        self.DECAY_STEP = DECAY_STEP
        self.BN_INIT_DECAY = 0.2
        self.BN_DECAY_DECAY_RATE = 0.5
        self.BN_DECAY_DECAY_STEP = float(self.DECAY_STEP)
        self.BN_DECAY_CLIP = 0.99
        self.LOG_FOUT = LOG_FOUT
        self.BASE_LEARNING_RATE = BASE_LEARNING_RATE
        self.DECAY_RATE = DECAY_RATE
        self.BATCH_SIZE = BATCH_SIZE

    def log_string(self, out_str):
        # self.LOG_FOUT.write(out_str+'\n')
        # self.LOG_FOUT.flush()
        print(out_str)

    def get_learning_rate(self, batch):
        learning_rate_base = tf.train.exponential_decay(
                            self.BASE_LEARNING_RATE,  # Base learning rate.
                            batch * self.BATCH_SIZE,  # Current index into the dataset.
                            self.DECAY_STEP,          # Decay step.
                            self.DECAY_RATE,          # Decay rate.
                            staircase=True)
        learning_rate = tf.maximum(learning_rate_base, 0.000001) # CLIP THE LEARNING RATE!
        return learning_rate


    def get_bn_decay(self, batch):
        bn_momentum = tf.train.exponential_decay(
                        self.BN_INIT_DECAY,
                        batch * self.BATCH_SIZE,
                        self.BN_DECAY_DECAY_STEP,
                        self.BN_DECAY_DECAY_RATE,
                        staircase=True)
        bn_decay = tf.minimum(self.BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay

    def get_batch(self, dataset, idxs, start_idx, end_idx):
        bsize = end_idx-start_idx
        batch_data = np.zeros((bsize, dataset.NUM_POINTS, 3))
        batch_data_shuffled = np.zeros((bsize, dataset.NUM_POINTS, 3))
        #area_data_shuffled = np.zeros((bsize, dataset.NUM_POINTS,dataset.NUM_POINTS))

        permutation_data = np.zeros((bsize, dataset.NUM_POINTS))

        for i in range(bsize):
            point_set_shuffled, point_set,permutation = dataset[idxs[i+start_idx]]
            batch_data[i,...] = point_set
            batch_data_shuffled[i,...] = point_set_shuffled
            #area_data_shuffled[i,...] = area_shuffled
            permutation_data[i,...] = permutation
        return batch_data_shuffled, batch_data, permutation_data
