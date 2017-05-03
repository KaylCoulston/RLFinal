# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import threading
import numpy as np

import signal
import random
import math
import os
import time
import pickle

from game_ac_network import GameACFFNetwork#, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import USE_LSTM
from constants import METHOD

import os.path

device = "/cpu:0"

initial_learning_rate = 0.0007 # suggested by https://gitter.im/traai/async-deep-rl
global_t = 0


if os.path.isfile("Rewards_onestep_SARSA_Binary.txt"):
	global_rewards = pickle.load(open("Rewards_onestep_SARSA_Binary.txt", 'rb'))
	os.remove("Rewards_onestep_SARSA_Binary.txt")
else:
	global_rewards = dict()



def signal_handler(signal, frame):
  global stop_requested
  print('You pressed Ctrl+C!')
  stop_requested = True

stop_requested = False

#Initialize Networks
global_network = GameACFFNetwork(ACTION_SIZE, -1, device)
global_network.prepare_loss()
target_network = GameACFFNetwork(ACTION_SIZE, -2, device)

training_threads = []

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = RMSP_EPSILON,
                              clip_norm = GRAD_NORM_CLIP,
                              device = device)

for i in range(PARALLEL_SIZE):
  training_thread = A3CTrainingThread(i, global_network, target_network, initial_learning_rate,
                                      learning_rate_input,
                                      grad_applier, MAX_TIME_STEP,
                                      device = device)
  training_threads.append(training_thread)

# prepare session V=[-0.08359856 -0.06895462 -0.07184759]
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

init = tf.global_variables_initializer()
sess.run(init)
sess.run(target_network.sync_from(global_network))
#TODO: Check sync
print("target network initialed with global network")

# summary for tensorboard
score_input = tf.placeholder(tf.int32)
ep_input = tf.placeholder(tf.float32)
lr_input = tf.placeholder(tf.float32)
value_input = tf.placeholder(tf.float32)
td_input = tf.placeholder(tf.float32)
tf.summary.scalar("score", score_input)
tf.summary.scalar("avgtd", td_input)
tf.summary.scalar("avgvalue", value_input)
tf.summary.scalar("epsilon", ep_input)
tf.summary.scalar("learn_rate", lr_input)

#TODO: Add testing function

summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

# init or load checkpoint with saver
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
  tokens = checkpoint.model_checkpoint_path.split("-")
  # set global step
  global_t = int(tokens[1])
  print(">>> global step set: ", global_t)
  # set wall time
  wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
  with open(wall_t_fname, 'r') as f:
    wall_t = float(f.read())
else:
  print("Could not find old checkpoint")
  # set wall time
  wall_t = 0.0


#i_target = 40000
#sync_target = target_network.sync_from(global_network)

def train_function(parallel_index):
  global global_t
  training_thread = training_threads[parallel_index]
  # set start_time
  start_time = time.time() - wall_t
  training_thread.set_start_time(start_time)

  while True:
    if stop_requested:
      break
    if global_t > MAX_TIME_STEP:
      break

    diff_t, episode_reward = training_thread.process(sess, global_t, summary_writer, summary_op, score_input, ep_input, lr_input, td_input, value_input, target_network, global_network)

    global_t += diff_t

    #for _ in range(diff_t):
    #	global_t += 1
    #	if global_t % i_target == 0:
    #		temp_global_t = global_t
    #		sess.run(sync_target)
    #		print("-----target network update at global time step {}-----".format(temp_global_t))
    #		print("Algorithm: {}".format(METHOD))
  
    #Add value to dictionary
    if global_t in global_rewards.keys():
    	global_rewards[global_t].append(episode_reward)
    else:
    	global_rewards[global_t] = [episode_reward]


#STARTS HERE!
train_threads = []
for i in range(PARALLEL_SIZE):
  train_threads.append(threading.Thread(target=train_function, args=(i,)))

signal.signal(signal.SIGINT, signal_handler)

# set start time
start_time = time.time() - wall_t

for t in train_threads:
  t.start()

print('Press Ctrl+C to stop')
signal.pause()

for t in train_threads:
  t.join()

if not os.path.exists(CHECKPOINT_DIR):
  os.mkdir(CHECKPOINT_DIR)

# write wall time
wall_t = time.time() - start_time
wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
with open(wall_t_fname, 'w') as f:
  f.write(str(wall_t))

print('Now saving data. Please wait')
pickle.dump(global_rewards, open("Rewards_onestep_SARSA_Binary.txt", 'wb'), protocol = pickle.HIGHEST_PROTOCOL)

saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)