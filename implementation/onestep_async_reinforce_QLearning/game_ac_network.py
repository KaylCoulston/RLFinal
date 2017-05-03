# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import numpy as np

# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(object):
  def __init__(self,
               action_size,
               thread_index, # -1 for global, -2 for target
               device="/cpu:0"):
    self._action_size = action_size
    self._thread_index = thread_index
    self._device = device

  def prepare_loss(self):
    with tf.device(self._device):
      # taken action (input for policy)
      self.a = tf.placeholder("float", [None, self._action_size])

      #comute Q(s_t, a_i)
      q_out = tf.reduce_sum( tf.multiply( self.qvalue, self.a ), reduction_indices=1 )

      # R (target_q)
      self.r = tf.placeholder("float", [None])

      # value loss (output)
      # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
      value_loss = tf.nn.l2_loss(self.r - q_out)
      #value_loss = tf.reduce_mean(tf.square(self.r - q_out))

      # gradienet of policy and value are summed up
      self.total_loss = value_loss

  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "GameACNetwork", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  def _fc_variable(self, weight_shape):
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  def _conv_variable(self, weight_shape):
    w = weight_shape[0]
    h = weight_shape[1]
    input_channels  = weight_shape[2]
    output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

# Actor-Critic FF Network
class GameACFFNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0"):
    GameACNetwork.__init__(self, action_size, thread_index, device)

    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      # state (input)
      self.s = tf.placeholder("float", [None, 84, 84, 4])

      self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16])  # stride=4
      self.W_fc1, self.b_fc1 = self._fc_variable([2592, 256])

      self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32]) # stride=2
      self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size])

      h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 4) + self.b_conv1)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

      h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
      h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

      # q-value (output)
      self.qvalue = tf.add(tf.matmul(h_fc1, self.W_fc2), self.b_fc2)

  def run_qvalue(self, sess, s_t):
    q_out = sess.run( self.qvalue, feed_dict = {self.s : [s_t]} )
    return q_out[0]

  def run_max_qtarget(self, sess, s_t):
    q_out = sess.run( self.qvalue, feed_dict = {self.s : [s_t]} )
    return np.max(q_out[0])

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2]