import tensorflow as tf
import math
import time
import numpy as np
import os
import pdb
import sys
from tf_utils import pointer_util
from tf_utils import edge_tf_util as tf_util
from sklearn.neighbors import NearestNeighbors
from config import cfg
from capsLayer import CapsLayer
from utils import *
epsilon = 1e-9


def input_placeholder(batch_size, num_point, n_classes = 10):
    pointclouds_pl = tf.placeholder(tf.float32,
                   shape=(batch_size, num_point, 3),name = 'pcl_placeholder')
    labels_pl = tf.placeholder(tf.int32,
                shape=(batch_size),name ='label_placeholder')
    print ('*********Training Pointer Capsnet v1***************************')
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, n_outputs, bn_decay=None):
    """ ConvNet baseline, input is BxNx3 gray image """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -2)

    k = 10
    nearest_pts_id = tf.py_func(pointer_util.get_nearest_neighbors_id,[input_image,k],tf.int32)
    # pointer_util.get_nearest_neighbors_id(input_image,k)
    # pdb.set_trace()
    nearest_pts_id = tf.reshape(nearest_pts_id, (batch_size, num_point,k))
    #pdb.set_trace()

    global_edge_features = tf.py_func(pointer_util.get_global_features,[input_image,nearest_pts_id,k],tf.float32)
    local_edge_features  = tf.py_func(pointer_util.get_local_features,[input_image,nearest_pts_id,k],tf.float32)

    global_edge_features = tf.reshape(global_edge_features, (batch_size,num_point,k,3))
    local_edge_features  = tf.reshape(local_edge_features, (batch_size,num_point,k,3))

    global_feature_1 = pointer_util.feature_network(global_edge_features,
                                                    mlp = [126],
                                                    name ='global_feature_1_',
                                                    is_training = is_training,
                                                    bn_decay = bn_decay)
    local_feature_1  = pointer_util.feature_network(local_edge_features,
                                                    mlp = [126],
                                                    name ='local_feature_1_',
                                                    is_training = is_training,
                                                    bn_decay = bn_decay)

    out_feature_1 = tf_util.conv2d(tf.concat([global_feature_1, local_feature_1], axis=-1),
                       126, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='out_feature_1', bn_decay=bn_decay, is_dist=True)

    out_feature_1 = tf.reduce_max(out_feature_1, axis = -2, keepdims = True)
    #shape (10,1000,1,126)

    primaryCaps = CapsLayer(num_outputs= 40 , vec_len=8, with_routing=False, layer_type='CONV')
    caps1 = primaryCaps(out_feature_1, kernel_size = 10, stride=2, scope = 'caps_layer_1')


    # DigitCaps layer, return shape [batch_size, 10, 16, 1]
    # with tf.variable_scope('DigitCaps_layer'):
    digitCaps = CapsLayer(num_outputs=n_outputs, vec_len=16, with_routing=True, layer_type='FC')
    caps2 = digitCaps(caps1)
    #caps 2 shape : 10x40x16x1

    v_length = tf.sqrt(reduce_sum(tf.square(caps2),
                                       axis=2, keepdims=True) + epsilon)

    softmax_v = softmax(v_length, axis=1)
    # assert self.softmax_v.get_shape() == [cfg.batch_size, self.num_label, 1, 1]

    # b). pick out the index of max softmax val of the 10 caps
    # [batch_size, 10, 1, 1] => [batch_size] (index)
    argmax_idx = tf.to_int32(tf.argmax(softmax_v, axis=1))
    # assert self.argmax_idx.get_shape() == [cfg.batch_size, 1, 1]
    # argmax_idx = tf.reshape(argmax_idx, shape=(batch_size, ))
    return tf.reshape(v_length, (batch_size, n_outputs))

def get_loss(v_length, label, batch_size):
    """ pred: B,40; label: B,N """
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    # # loss = tf.nn.weighted_cross_entropy_with_logits(targets = label, logits = pred,
    # #           pos_weight = np.array([1.0,60.0]))
    # return tf.reduce_mean(loss)
    #
    # [batch_size, 10, 1, 1]
    # max_l = max(0, m_plus-||v_c||)^2
    # pdb.set_trace()
    max_l = tf.square(tf.maximum(0., cfg.m_plus - v_length))
    # max_r = max(0, ||v_c||-m_minus)^2
    max_r = tf.square(tf.maximum(0., v_length - cfg.m_minus))
    # assert max_l.get_shape() == [cfg.batch_size, self.num_label, 1, 1]
    # pdb.set_trace()
    # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
    max_l = tf.reshape(max_l, shape=(batch_size, -1))
    max_r = tf.reshape(max_r, shape=(batch_size, -1))
    # calc T_c: [batch_size, 10]
    # T_c = Y, is my understanding correct? Try it.
    T_c = tf.one_hot(label, depth = 40)
    # [batch_size, 10], element-wise multiply
    L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r
    margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
    return margin_loss

if __name__=='__main__':
    pcl, label = input_placeholder(10,1000)
    # is_train = tf.placeholder(tf.bool, shape= ())
    pcl_h  = np.zeros((10,1000,4))
    is_train = tf.placeholder(tf.bool, shape =())
    model_pred, vlen = get_model(pcl, is_train, 40)
    loss = get_loss(model_pred, label, batch_size = 10)
    pdb.set_trace()
    print('efwjf')
