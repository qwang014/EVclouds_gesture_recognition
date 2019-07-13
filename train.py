"""
Modified from: https://github.com/charlesq34/pointnet/blob/master/train.py
Author: Wang Qinyi
Date: July 2018
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls_basic', help='Model name [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')

parser.add_argument('--num_classes', type=int, default=10, help='10 getures + 1 random')
parser.add_argument('--window_size', type=float, default=0.5, help='second')
parser.add_argument('--step_size', type=float, default=0.25, help='second')
parser.add_argument('--seq_len', type=int, default=1, help='sequence length [default: 1]')
parser.add_argument('--num_events', type=int, default=512, help='number of events in a sliding window [256/512/1024]')

parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model)
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) 
os.system('cp train.py %s' % (LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = FLAGS.num_classes
WINDOW_SIZE = FLAGS.window_size
STEP_SIZE = FLAGS.step_size
SEQ_LEN = FLAGS.seq_len
NUM_EVENTS = FLAGS.num_events

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
DUMP_DIR = FLAGS.dump_dir
BASE_LEARNING_RATE = FLAGS.learning_rate

MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate


GESTURE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/gesture_names.txt'))] 


BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
                eventclouds, labels = MODEL.placeholder_inputs(batch_size = BATCH_SIZE, 
                                                               seq_len = SEQ_LEN,
                                                               num_point = NUM_EVENTS)
                is_training = tf.placeholder(tf.bool, shape=())
                print(is_training)


                batch = tf.Variable(0)
                bn_decay = get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)
            
                # Get model and loss 
                pred = MODEL.get_model(eventclouds = eventclouds,
                                       seq_len = SEQ_LEN,
                                       num_classes = NUM_CLASSES,
                                       is_training = is_training,
                                       bn_decay = bn_decay)
            
                loss = MODEL.get_loss(pred, labels)
                tf.summary.scalar('loss', loss)

                correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels))
                accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
                tf.summary.scalar('accuracy', accuracy)

                # Get training operator
                learning_rate = get_learning_rate(batch)
                tf.summary.scalar('learning_rate', learning_rate)
                if OPTIMIZER == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(loss, global_step=batch)
            
                # Add ops to save and restore all the variables.
                saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
   
        sess.run(init, {is_training: True})

        ops = {'eventclouds': eventclouds,
               'labels': labels,
               'is_training': is_training,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch_1(sess, ops, test_writer)
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    
    
    for fn in range(len(TRAIN_FILES)):
        log_string('----train file ' + str(fn) + '-----')
        print(TRAIN_FILES[train_file_idxs[fn]])
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,:,0:NUM_EVENTS,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        #-------------shuffle data---------------------------
        idx = np.arange(file_size)
        np.random.shuffle(idx)
        current_data = current_data[idx,...]
        current_label = current_label[idx,...]
        #-----------------------------------------------------

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
           
            batch_data = current_data[start_idx:end_idx,...]
            batch_label = current_label[start_idx:end_idx,...]
        
            # print('batch_label',batch_label)
            feed_dict = {ops['eventclouds']: batch_data,
                         ops['labels']: batch_label,
                         ops['is_training']: is_training}
            
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            
            train_writer.add_summary(summary, step)
            
            pred_val = np.argmax(pred_val, 1)
            # print('pred_val',pred_val)
            correct = np.sum(pred_val == batch_label)
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val

        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch_1(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    for fn in range(len(TEST_FILES)):
        log_string('----test file ' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,:,0:NUM_EVENTS,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

       
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE

            batch_data = current_data[start_idx:end_idx,...]
            batch_label = current_label[start_idx:end_idx,...]
            
            feed_dict = {ops['eventclouds']: batch_data,
                         ops['labels']: batch_label,
                         ops['is_training']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            
            test_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == batch_label)
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val

            for i in range(BATCH_SIZE):
                l = batch_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i] == l)
                fout.write('%d, %d\n' % (pred_val[i], l))
            
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    for i in range(NUM_CLASSES):
        name = GESTURE_NAMES[i]
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
    

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
