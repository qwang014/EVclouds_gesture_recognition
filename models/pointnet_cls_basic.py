
"""
Modified from: https://github.com/charlesq34/pointnet/blob/master/models/pointnet_cls_basic.py
Author: Wang Qinyi
Date: Jan 2018
"""
import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

def placeholder_inputs(batch_size,seq_len,num_point):
    eventclouds = tf.placeholder(tf.float32, shape=(batch_size,seq_len, num_point, 3))
    labels = tf.placeholder(tf.int32, shape=(batch_size))
    return eventclouds, labels


def get_model(eventclouds, 
              seq_len,
              num_classes,
              is_training,
              bn_decay=None):
    """ input is B x S x N x 3, output B x num_classes """
    num_point = eventclouds.get_shape()[-2].value 
    batch_size = eventclouds.get_shape()[0].value 
    eventclouds = tf.reshape(eventclouds, [-1, num_point, 3])
    eventclouds = tf.expand_dims(eventclouds, -1)
   
    net = tf_util.conv2d(eventclouds, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')
    
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, num_classes, activation_fn=None, scope='fc3')
    print(net.shape)
    return net


def get_loss(pred, label):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1,1024,3))
        outputs = get_model(inputs,1,11,tf.constant(True))
        print(outputs)
