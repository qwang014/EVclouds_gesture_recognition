"""
Modified from: https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_cls_ssg.py
Author: Wang Qinyi
Date: Jan 2018
"""
import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module


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
    
    end_points = {}
    l0_xyz = eventclouds
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Set abstraction layers
   
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=256, radius=0.2, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=64, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')
   
    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, num_classes, activation_fn=None, scope='fc3')

    return net


def get_loss(pred, label):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1,1024,3))
        outputs = get_model(inputs,1,11,tf.constant(True))
        print(outputs)
