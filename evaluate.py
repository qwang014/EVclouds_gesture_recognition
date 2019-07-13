"""
Modified from: https://github.com/charlesq34/pointnet/blob/master/evaluate.py
Author: Wang Qinyi
Date: Jan 2018
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import timeit


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls_basic', help='Model name pointnet_cls_basic or pointnet2_cls_ssg')

parser.add_argument('--num_classes', type=int, default=10, help='10 getures + 1 random')
parser.add_argument('--window_size', type=float, default=0.5, help='second')
parser.add_argument('--step_size', type=float, default=0.25, help='second')
parser.add_argument('--seq_len', type=int, default=1, help='sequence length [default: 1]')
parser.add_argument('--num_events', type=int, default=512, help='number of events in a sliding window [256/512/1024]')

parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 1]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model)

NUM_CLASSES = FLAGS.num_classes
WINDOW_SIZE = FLAGS.window_size
STEP_SIZE = FLAGS.step_size
SEQ_LEN = FLAGS.seq_len
NUM_EVENTS = FLAGS.num_events

BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
DUMP_DIR = FLAGS.dump_dir

if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


GESTURE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/gesture_names.txt'))] 

HOSTNAME = socket.gethostname()

TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        eventclouds, labels = MODEL.placeholder_inputs(batch_size = BATCH_SIZE, 
                                                       seq_len = SEQ_LEN,
                                                       num_point = NUM_EVENTS)
              
        is_training = tf.placeholder(tf.bool, shape=())

        # simple model
        pred = MODEL.get_model(eventclouds = eventclouds,
                               seq_len = SEQ_LEN,
                               num_classes = NUM_CLASSES,
                               is_training = is_training)
        
        loss = MODEL.get_loss(pred, labels)
       
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    saver.restore(sess, MODEL_PATH)
    
    
    log_string("Model restored.")

    ops = {'eventclouds': eventclouds,
           'labels': labels,
           'is_training': is_training,
           'pred': pred,
           'loss': loss}

    eval_one_epoch(sess, ops)
    

   
def eval_one_epoch(sess, ops):
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
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
            
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


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()
