"""
Actor-Critic with continuous action using TD-error as the Advantage, Reinforcement Learning.

The Pendulum example (based on https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb)

Cannot converge!!! oscillate!!!

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow r1.3
gym 0.8.0
"""


import tensorflow as tf
import numpy as np
import gym

np.random.seed(2)
tf.set_random_seed(2)  # reproducible



class Actor(object):
    def __init__(self, sess, n_features, n_actions, action_bounds, lr=0.0001):
        self.sess = sess
        self.action_bounds = (action_bounds[0],action_bounds[1])
        self.action_dim = n_actions

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.float32, [n_actions], name="act")
        self.td_error = tf.placeholder(
            tf.float32, None, name="td_error")  # TD_error

        l1 = tf.layers.dense(
            inputs=self.s,
            units=100,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l1'
        )

        l2 = tf.layers.dense(
            inputs=l1,
            units=40,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l2'
        )

        mu = tf.layers.dense(
            inputs=l2,
            units=n_actions,  # output units
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='mu'
        )

        sigma = tf.layers.dense(
            inputs=l2,
            units=n_actions,  # output units
            activation=tf.nn.softplus,  # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(1.),  # biases
            name='sigma'
        )
        global_step = tf.Variable(0, trainable=False)
        # self.e = epsilon = tf.train.exponential_decay(2., global_step, 1000, 0.9)

        self.mu, self.sigma = tf.squeeze(mu), tf.squeeze(sigma)

        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        self.action = tf.squeeze(tf.clip_by_value(
            self.normal_dist.sample(1)*3e-1, self.action_bounds[0], self.action_bounds[1]))

        with tf.name_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(
                self.a)  # loss without advantage
            # advantage (TD_error) guided loss
            self.exp_v = log_prob * self.td_error
            # Add cross entropy cost to encourage exploration
            self.exp_v += 0.01*self.normal_dist.entropy()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(
                lr).minimize(-self.exp_v, global_step)  # max(v) = min(-v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        # a = a[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        # get probabilities for all actions
        a = self.sess.run(self.action, {self.s: s})
        return a


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01, GAMMA=0.9):
        self.gamma = GAMMA
        self.sess = sess
        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32, [1, n_features], "state")
            self.v_ = tf.placeholder(tf.float32, [1, 1], name="v_next")
            self.r = tf.placeholder(tf.float32, name='r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=100,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )
            l2 = tf.layers.dense(
                inputs=l1,
                units=40,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )

            self.v = tf.layers.dense(
                inputs=l2,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(
                self.r + self.gamma * self.v_ - self.v)
            # TD_error = (r+gamma*V_next) - V_eval
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error



if __name__ == "__main__":

    OUTPUT_GRAPH = False
    MAX_EPISODE = 1000
    MAX_EP_STEPS = 200
    # renders environment if total episode reward is greater then this threshold
    DISPLAY_REWARD_THRESHOLD = -100
    RENDER = False  # rendering wastes time
    GAMMA = 0.9
    LR_A = 0.001    # learning rate for actor
    LR_C = 0.01     # learning rate for critic

        
    env = gym.make('Pendulum-v0')
    env.seed(1)  # reproducible
    env = env.unwrapped

    N_S = env.observation_space.shape[0]
    A_BOUND = env.action_space.high

    sess = tf.Session()

    actor = Actor(sess, n_features=N_S,n_actions=1, lr=LR_A,
                  action_bounds=[-A_BOUND, A_BOUND])
    critic = Critic(sess, n_features=N_S, lr=LR_C)

    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        t = 0
        ep_rs = []
        while True:
            # if RENDER:
            env.render()
            a = actor.choose_action(s)
            print(a)
            s_, r, done, info = env.step([a])
            r /= 10

            # gradient = grad[r + gamma * V(s_) - V(s)]
            td_error = critic.learn(s, r, s_)
            # true_gradient = grad[logPi(s,a) * td_error]
            actor.learn(s, [a], td_error)

            s = s_
            t += 1
            ep_rs.append(r)
            if t > MAX_EP_STEPS:
                ep_rs_sum = sum(ep_rs)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.9 + ep_rs_sum * 0.1
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True  # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))
                break
