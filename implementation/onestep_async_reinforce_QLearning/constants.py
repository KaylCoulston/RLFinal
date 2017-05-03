# -*- coding: utf-8 -*-
from ale_python_interface import ALEInterface

LOCAL_T_MAX = 20 # repeat step size, try with minibatch 32 as in DQN
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints'
LOG_FILE = 'tmp/a3c_log'
INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate

PARALLEL_SIZE = 16 # parallel thread size
ROM = "pong.bin"     # action size = 3
_ale = ALEInterface()
_ale.loadROM(ROM.encode('ascii'))
ACTION_SIZE = len(_ale.getMinimalActionSet()) # auto-get action size

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 0.01 # entropy regurarlization constant
MAX_TIME_STEP = 20 * 10**7
GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = False # To use GPU, set True
USE_LSTM = False # True for A3C LSTM, False for A3C FF
TRAIN_EPSILON_START = 1.0 # esilon parameter for e-greedy
TRAIN_EPSILON_PROBABILITY = [.4, .3, .3]
TRAIN_EPSILON_END = [.1, .01, .5]
METHOD = "oneq" #choose RL algorithms, default one-step q, other options: sarsa
