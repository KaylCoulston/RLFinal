# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import numpy as np
import random
import time
import sys

from game_state import GameState
from game_state import ACTION_SIZE
from game_ac_network import GameACFFNetwork#, GameACLSTMNetwork

from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import USE_LSTM
from constants import TRAIN_EPSILON_START
from constants import TRAIN_EPSILON_PROBABILITY
from constants import TRAIN_EPSILON_END
from constants import PARALLEL_SIZE
from constants import METHOD

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 10000

class A3CTrainingThread(object):
  def __init__(self,
               thread_index,
               global_network,
               target_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device):

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step
    # self.file = open("Rewards{}.txt".format(thread_index + 1), 'w')
    self.global_reward = dict()


    with tf.device(device):
      var_refs = [v._ref() for v in global_network.get_vars()]
      self.gradients = tf.gradients(
        global_network.total_loss, var_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)

    self.apply_gradients = grad_applier.apply_gradients(
      global_network.get_vars(),
      self.gradients )

    self.sync_target = target_network.sync_from(global_network)

    self.game_state = GameState(113 * thread_index)

    self.local_t = 0
    self.global_local_t = 0

    self.initial_learning_rate = initial_learning_rate

    # variable controling log output
    self.prev_local_t = 0

    self.episode_reward = 0


    self.train_epsilon_start = TRAIN_EPSILON_START
    self.train_epsilon_end = np.random.choice(a=TRAIN_EPSILON_END, p=TRAIN_EPSILON_PROBABILITY)

  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    #print learning_rate
    return learning_rate

  def _epsilon_decay(self, global_time_step):
    #TODO: Try using 1MIL next
    max_decay_time_step = 4000000 #as in DQN
    epsilon_decay = 1.0 - global_time_step * ((1.0 - self.train_epsilon_end) / max_decay_time_step)
    if epsilon_decay < self.train_epsilon_end:
      epsilon_decay = self.train_epsilon_end
    return epsilon_decay

  def choose_action(self, q_values, train_epsilon):
    if np.random.random() < train_epsilon:
      return random.randint(0, ACTION_SIZE-1)
    return np.argmax(q_values)

  #sess, summary_writer, summary_op, score_input, ep_input, lr_input, td_input, value_input,
  #record_episode_reward, self.sample_ep, cur_learning_rate, avg_td, avg_value, global_t
  def _record_score(self, sess, summary_writer, summary_op, score_input, ep_input, lr_input, td_input, value_input,
                    score, ep, lr, td, value, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      score_input: score,
      td_input: td,
      value_input: value,
      ep_input: ep,
      lr_input: lr}
      )
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()

  def set_start_time(self, start_time):
    self.start_time = start_time

  def process(self, sess, global_t, summary_writer, summary_op, score_input, ep_input, lr_input, td_input, value_input, target_network, global_network):
    states = []
    actions = []
    return_value = []

    td_arr = []

    record_episode_reward = 0

    terminal_end = False

    local_start_t = self.local_t

    # copy weights from shared to local
    #sess.run( self.sync )

    # t_max times loop = 20
    for i in range(LOCAL_T_MAX):
      qvalue = global_network.run_qvalue(sess, self.game_state.s_t)
      self.sample_ep = self._epsilon_decay(global_t)
      action = self.choose_action(qvalue, self.sample_ep)

      states.append(self.game_state.s_t)
      actions.append(action)

      # process game
      self.game_state.process(action)

      # receive game result
      reward = self.game_state.reward
      terminal = self.game_state.terminal

      if terminal:
        y = np.clip(reward, -1, 1)
      else:
        q_next = target_network.run_qvalue(sess, self.game_state.s_t1)
        y = np.clip(reward, -1, 1) + GAMMA * np.max(q_next)

        #if METHOD == "sarsa":
        #  next_action = self.choose_action(q_next, self.sample_ep)
        #  y = np.clip(reward, -1, 1) + GAMMA * q_next[next_action]
        
        #elif METHOD == "oneq":
        #  y = np.clip(reward, -1, 1) + GAMMA * np.max(q_next)

      #accumulate TD error for the action took
      td_arr.append((y - qvalue[action]) **2)

      self.episode_reward += reward

      #This goes to gradient computation
      return_value.append(y)

      # s_t1 -> s_t
      self.game_state.update()

      self.local_t += 1
      global_t += 1

      #sync target network from global every 40000 steps
      i_target = 40000
      if global_t % i_target ==  0:
        sess.run(self.sync_target)
        print("Updating Target Network!")
        print("===== target network updated at global step {}".format(global_t))

      if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
        #print("pi={}".format(pi_))
        print(" V={}".format(qvalue))

      if terminal:
        terminal_end = True
        print("score={}".format(self.episode_reward))
        record_episode_reward = self.episode_reward

        self.episode_reward = 0
        self.game_state.reset()

        break

    batch_a = []
    for ai in actions:
      a = np.zeros([ACTION_SIZE])
      a[ai] = 1
      batch_a.append(a)

    cur_learning_rate = self._anneal_learning_rate(global_t)

    avg_td = np.average(td_arr)
    avg_value = np.average(return_value)


    sess.run( self.apply_gradients,
              feed_dict = {
                global_network.s: states,
                global_network.a: batch_a,
                global_network.r: return_value,
                self.learning_rate_input: cur_learning_rate} )

    if terminal_end:
      self._record_score(sess, summary_writer, summary_op, score_input, ep_input, lr_input, td_input, value_input,
                         record_episode_reward, self.sample_ep, cur_learning_rate, avg_td, avg_value, global_t)
      
      #save the reward to the file
      #self.file.write("{}\n".format(record_episode_reward))


    if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
        global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))


    local_dif_t = self.local_t - local_start_t
    return [local_dif_t, record_episode_reward]
