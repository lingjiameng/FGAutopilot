"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time
from DRL.replay_memory.prioritized_replay_memory import PrioritizedReplayMemory


###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, MEMORY_CAPACITY=16384, LR_A=0.001, LR_C=0.002):
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.TAU = 0.01      # soft replacement
        self.GAMMA = 0.9     # reward discount
        self.BATCH_SIZE = 64
        self.LR_A = LR_A    # learning rate for actor
        self.LR_C = LR_C    # learning rate for critic
        self.sample_alpha = 0.7

        self.td_errors = np.array([])
        self.prois = np.array([])
        # self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)

        self.proi = PrioritizedReplayMemory(MEMORY_CAPACITY, alpha=0.7)

        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.weights = tf.placeholder(tf.float32, [None, 1], "w")
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(
            decay=1 - self.TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(
            c_params)]      # soft update operation
        # replaced target parameters
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(
            self.LR_A).minimize(a_loss, var_list=a_params)

        # soft replacement happened at here
        with tf.control_dependencies(target_update):
            q_target = self.R + self.GAMMA * q_
            td_error = tf.losses.mean_squared_error(
                labels=q_target, predictions=q, weights=self.weights)
            self.ctrain = tf.train.AdamOptimizer(
                self.LR_C).minimize(td_error, var_list=c_params)
            # self._opt_c(td_error,c_params,q_target,q)

        self.get_td_error = self._get_td_error(self.R + self.GAMMA * q_ - q)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        batch = self.proi.get_random_sample(self.BATCH_SIZE)
        batch = np.array(batch).T

        bt = np.array([var for var in batch[3]])
        # print(type(bt),bt.shape,"\n",bt)
        weights = batch[2]
        proi_ind = batch[0]
        prois = batch[1]
        # print(indices,type(indices))
        # indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        # bt = self.memory[indices, :]

        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        # print(bs)
        w = weights[:, np.newaxis]
        # print(w.shape)

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {
                      self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.weights: w})
        td_error = self.sess.run(
            self.get_td_error, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        # print(td_error)
        self.td_errors = np.append(self.td_errors, td_error)
        self.prois = np.append(self.prois, prois)

        for ind, error in zip(proi_ind, abs(td_error)):
            self.proi.update(ind, abs(error))

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # replace the old memory with new memory
        index = self.pointer % self.MEMORY_CAPACITY
        # self.memory[index, :] = transition

        self.proi.add(transition)

        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(
                s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(
                net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable(
                'w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable(
                'w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    def _get_td_error(self, td_error):
        return tf.squeeze(td_error)


###############################  training  ####################################
def main():
    MAX_EPISODES = 200
    MAX_EP_STEPS = 200
    RENDER = False
    ENV_NAME = 'Pendulum-v0'
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(a_dim, s_dim, a_bound, LR_A=0.001, LR_C=0.002)

    var = 3  # control exploration
    t1 = time.time()

    ep_rewards = []

    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            # add randomness to action selection for exploration
            a = np.clip(np.random.normal(a, var), -2, 2)
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r / 10, s_)

            if ddpg.pointer > ddpg.MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: %i' %
                      int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward > -300:
                    RENDER = True
                # if i > 180:
                   # RENDER  = True
                break

        ep_rewards.append(ep_reward)

    print('Running time: ', time.time() - t1)
    print("---------------")
    print(ddpg.td_errors.min())
    print(ddpg.td_errors.max())
    print(ddpg.td_errors.mean())
    print(ddpg.td_errors.std())
    print("---------------")
    print(ddpg.prois.min())
    print(ddpg.prois.max())
    print(ddpg.prois.mean())
    print(ddpg.prois.std())

    import matplotlib.pyplot as plt
    plt.plot(ep_rewards)
    plt.show()

    '''
    ---------------
    -1.7622344493865967
    2.2921931743621826
    -0.5278277019309097
    0.7084490109292628
    ---------------
    0.0018760661498405885
    1.6449429588472397
    0.883099566969316
    0.27711443596732577
    '''
