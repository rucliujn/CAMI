from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import range# pylint: disable=redefined-builtin
from six.moves import zip	 # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
import math
import os
import random
import sys
import time

import numpy as np
from six.moves import range# pylint: disable=redefined-builtin
import tensorflow as tf

def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)
																				
def get_scores(model, example_multi_vec, product_vec, product_bias, user_lambda, user_p, selection, user_cate, product_cate, query_att):
    #user_cate [batch * K, embed_size]
    #product_cate [product_size, embed_size]
    #
    #example_multi_vec [batch * K, embed_size]
    #product_vec [product_size, embed_size]
    #product_bias [product_size]
    product_size = array_ops.shape(product_vec)[0]
    user_lambda = tf.tile(tf.expand_dims(user_lambda, 1), [1, product_size])
    query_att = tf.tile(tf.expand_dims(query_att, 2), [1, 1, product_size])
    if (selection == 'soft') :
        match_score = tf.reshape(tf.matmul(example_multi_vec, product_vec, transpose_b=True),[-1, model.mult_size, product_size])#batch, K, product_size
        cate_score = tf.reshape(tf.matmul(user_cate, product_cate, transpose_b=True), [-1, model.mult_size, product_size])#batch, K, product_size
        user_p = tf.reshape(tf.tile(tf.expand_dims(user_p, 1),  [1, model.mult_size * product_size]), [-1, model.mult_size, product_size])
        attention = cate_score * query_att / model.tau
        #attention = cate_score * query_att / user_p for CAMI-p
        attention = attention - tf.tile(tf.reduce_max(attention, 1, keep_dims=True), [1, model.mult_size, 1])
        attention = tf.exp(attention) / tf.tile(tf.reduce_sum(tf.exp(attention), 1, keep_dims=True), [1, model.mult_size, 1])
        return user_lambda * tf.reduce_sum(attention * match_score, axis=1, keep_dims=False) + product_bias
    if (selection == 'gumbel') :
        match_score = tf.reshape(tf.matmul(example_multi_vec, product_vec, transpose_b=True), [-1, model.mult_size, product_size])#batch, K, product_size
        cate_score = tf.reshape(tf.matmul(user_cate, product_cate, transpose_b=True), [-1, model.mult_size, product_size])#batch, K, product_size
        cate_score = tf.nn.softmax(cate_score, dim=1)#batch, K, product_size
        score = match_score * cate_score
        distribution = tf.nn.softmax(score, dim=1)#batch, K, product_size
        gumbel_distribution = sample_gumbel(tf.shape(distribution))
        attention = tf.nn.softmax((tf.log(distribution) + gumbel_distribution) / 0.05, dim=1)
        return user_lambda * tf.reduce_sum(attention * score, axis=1, keep_dims=False) + product_bias 

def get_true_scores(model, example_multi_vec, product_vec, product_bias, user_lambda, user_p, selection, user_cate, product_cate, query_att):
    #user_cate [batch, K, embed_size]
    #product_cate [batch_size, embed_size]

    #example_multi_vec [batch, K, embed_size]
    #product_vec [batch_size, embed_size]
    #product_bias [batch_size]
    if (selection == 'soft') :
        match_score = tf.squeeze(tf.matmul(example_multi_vec, tf.expand_dims(product_vec, 2)), 2)#batch, K
        cate_score = tf.squeeze(tf.matmul(user_cate, tf.expand_dims(product_cate, 2)), 2)#batch, K
        user_p = tf.tile(tf.expand_dims(user_p, 1), [1, model.mult_size])
        attention = cate_score * query_att / model.tau 
        #attention = cate_score * query_att / user_p for CAMI-p
        attention = attention - tf.tile(tf.reduce_max(attention, 1, keep_dims=True), [1, model.mult_size])
        attention = tf.exp(attention) / tf.tile(tf.reduce_sum(tf.exp(attention), 1, keep_dims=True), [1, model.mult_size])
        return user_lambda * tf.reduce_sum(attention * match_score, axis=1) + product_bias
    if (selection == 'gumbel') :
        match_score = tf.squeeze(tf.matmul(example_multi_vec, tf.expand_dims(product_vec, 2)), 2)#batch, K
        cate_score = tf.squeeze(tf.matmul(user_cate, tf.expand_dims(product_cate, 2)), 2)#batch, K
        cate_score = tf.nn.softmax(cate_score, dim=1)#batch, K
        score = match_score * cate_score
        distribution = tf.nn.softmax(score, dim=1)#batch, K
        gumbel_distribution = sample_gumbel(tf.shape(distribution))
        attention = tf.nn.softmax((tf.log(distribution) + gumbel_distribution) / 0.05, dim=1)#batch, K
        return user_lambda * tf.reduce_sum(attention * score, axis=1) + product_bias