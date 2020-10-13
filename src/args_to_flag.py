
import os

def args_to_flag(parser, train=True):
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--model', default='model', help='Model name [default: model]')
    parser.add_argument('--log_dir', default='pretrained_models/log_model_custom', help='Log dir ')
    parser.add_argument('--model_path', default='pretrained_models/log_model/best_model_epoch_4427.ckpt', help='path to model')
    parser.add_argument('--num_point', type=int, default=1000, help='Point Number [default: 2048]')
    parser.add_argument('--max_epoch', type=int, default=6000, help='Epoch to run [default: 201]')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 32]')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
    parser.add_argument('--decay_step', type=int, default=8*1250*10, help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--no_rotation', action='store_true',default = False, help='Disable random rotation during training.')
    FLAGS = parser.parse_args()
    BATCH_SIZE = FLAGS.batch_size
    NUM_POINT = FLAGS.num_point
    MAX_EPOCH = FLAGS.max_epoch
    BASE_LEARNING_RATE = FLAGS.learning_rate
    GPU_INDEX = FLAGS.gpu
    MOMENTUM = FLAGS.momentum
    OPTIMIZER = FLAGS.optimizer
    DECAY_STEP = FLAGS.decay_step
    DECAY_RATE = FLAGS.decay_rate
    PATH_MODEL = FLAGS.model
    LOG_DIR = FLAGS.log_dir
    MODEL_PATH = FLAGS.model_path
    NO_ROTATION = FLAGS.no_rotation
    if train and not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
        LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train' + '.txt'), 'w')
        LOG_FOUT.write(str(FLAGS)+'\n')
    else:
        LOG_FOUT = ''
    return PATH_MODEL, LOG_DIR, BATCH_SIZE, NUM_POINT, MAX_EPOCH, BASE_LEARNING_RATE, GPU_INDEX, MOMENTUM, OPTIMIZER, DECAY_STEP, DECAY_RATE, LOG_FOUT, NO_ROTATION, MODEL_PATH
