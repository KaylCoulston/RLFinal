from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pandas as pd

#env = gym.make('FrozenLake-v0')
env = gym.make('Taxi-v2')
nstate  = env.observation_space.n
naction = env.action_space.n
print nstate
print naction
time.sleep(2)
#env = gym.make('Pong-v0')

#implement a simple Q network using tf
tf.reset_default_graph()

#feedforward
inputs1 = tf.placeholder(shape=[1,nstate],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([nstate, naction], 0, 0.01), name="weight")
Qout = tf.matmul(inputs1, W)#input state multiply weights
predict = tf.argmax(Qout, 1)

#loss: sum of MSE of target and predict Q for branch
# similar to V(s') - V(s)
nextQ = tf.placeholder(shape=[1, naction],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))#reduce_sum computes elementwise sum
trainer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)#more optimizer in tf
updateModel = trainer.minimize(loss)

# train the network
# tf needs variable initializer to feed value
init = tf.global_variables_initializer()

gamma = 0.9
epsilon = 0.1
num_episodes = 2000
totalreward = []

#start a session
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        #time.sleep(2)
        state = env.reset()
        for t in range(100):
            #env.render()
            action, targetQ = sess.run([predict, Qout],feed_dict={inputs1:np.identity(nstate)[state:state+1]})
            #print action
            #print targetQ
            if np.random.rand() < epsilon:#e-greedy
                action[0] = env.action_space.sample()
            #get next state
            nextstate, reward, terminal, _ = env.step(action[0])
            #get predict Q
            Qnext = sess.run(Qout, feed_dict={inputs1:np.identity(nstate)[nextstate:nextstate+1]})
            maxQ = np.max(Qnext)
            targetQ[0,action[0]] = reward + gamma * maxQ
            #update weights
            _, W1 = sess.run([updateModel, W], feed_dict={inputs1:np.identity(nstate)[state:state+1],nextQ:targetQ})
            state = nextstate
            if terminal:
                #env.render()
                print("episode ends at %d steps"%t)
                break

        #do a test after one episode
        R = 0
        state = env.reset()
        for t in range(100):
            #env.render()
            action = sess.run(predict,feed_dict={inputs1:np.identity(nstate)[state:state+1]})
            #get next state
            nextstate, reward, terminal, _ = env.step(action[0])
            state = nextstate
            R+=reward
            #time.sleep(0.5)
            if terminal:
                #env.render()
                #time.sleep(.5)
                print("episode ends at %d steps"%t)
                break
        totalreward.append(R)
plt.plot(pd.rolling_sum(pd.DataFrame(totalreward), 10), label="Q-NN")
plt.show()
