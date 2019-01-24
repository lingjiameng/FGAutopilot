
import DRL.ActorCritic as AC
import fgmodule.fgenv as fgenv

import tensorflow as tf
import numpy as np
import gym.spaces as gymspaces

LLC_FEATURES = [
    'pitch-deg',  # 飞机俯仰角
    'roll-deg',  # 飞机滚转角
    'heading-deg',  # 飞机朝向
    'vsi-fpm',  # 爬升速度
    'uBody-fps',  # 飞机沿机身X轴的速度
    'vBody-fps',  # 飞机沿机身Y轴的速度
    'wBody-fps',  # 飞机沿机身Z轴的速度
    'x-accel-fps_sec',  # 飞机沿机身X轴的加速度
    'y-accel-fps_sec',  # 飞机沿机身Y轴的加速度
    'z-accel-fps_sec',  # 飞机沿机身z轴的加速度
]

LLC_GOALS = {
    'pitch-deg': 0.0,  # 飞机俯仰角
    'roll-deg': 0.0,  # 飞机滚转角
    'heading-deg': 90.0,  # 飞机朝向
    'vsi-fpm': 0.0,  # 爬升速度
    'uBody-fps': 120.0,  # 飞机沿机身X轴的速度
    'vBody-fps': 0.0,  # 飞机沿机身Y轴的速度
    'wBody-fps': 0.0,  # 飞机沿机身Z轴的速度
    'x-accel-fps_sec': 5.0,  # 飞机沿机身X轴的加速度
    'y-accel-fps_sec': 0.0,  # 飞机沿机身Y轴的加速度
    'z-accel-fps_sec': 0.0,  # 飞机沿机身z轴的加速度
}

LLC_ACTIONS = [
    'aileron',  # 副翼 控制飞机翻滚 [-1,1]
    'elevator',  # 升降舵 控制飞机爬升 [-1,1]
    'rudder',  # 方向舵 控制飞机转弯（地面飞机方向控制） [-1,1]
    'throttle0',  # 油门0 [0,1]
    'throttle1'  # 油门1 [0,1]
    # 'flaps',  # 襟翼 在飞机起降过程中增加升力，阻力 [0,1],实测影响不大，而且有速度限制
    #TODO: 方向舵调整片
]
LLC_ACTION_BOUNDS = {
    'aileron': [-1, 1],  # 副翼 控制飞机翻滚 [-1,1] left/right
    'elevator': [-1, 1],  # 升降舵 控制飞机爬升 [-1,1] up/down
    'rudder': [-1, 1],  # 方向舵 控制飞机转弯（地面飞机方向控制） 0 /enter
    'throttle0': [0, 1],  # 油门0
    'throttle1': [0, 1],  # 油门1
    # 'flaps': [0, 0]  # 襟翼 在飞机起降过程中增加升力，阻力  Key[ / ]	Extend / retract flaps
    #TODO: 方向舵调整片
}


MAX_EPISODE = 1000
MAX_EP_STEPS = 200

GAMMA = 0.9
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic


def get_lower_upper_bound(bounds):
    '''
    format bound dict to lower and upper array
    '''
    lower = []
    upper = []
    for key in bounds.keys():
        lower.append(bounds[key][0])
        upper.append(bounds[key][1])
    return np.array(lower,dtype=np.float), np.array(upper,dtype=np.float)


class LLC():
    def __init__(self, n_features, n_actions, n_action_bounds, actorlr=0.0001, criticlr=0.01):

        self.sess = tf.Session()
        self.actor = AC.Actor(self.sess, n_features,
                              n_actions, n_action_bounds, actorlr)
        self.critic = AC.Critic(self.sess, n_features, criticlr)
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, state):
        '''
        input: 
            dict
        output:
            # control frame
            # [ % f, % f, % f, % f, % f\n]
            # aileron, elevator, rudder, throttle0, throttle1
            # 副翼, 升降舵, 方向舵, 油门0, 油门1
        '''

        action = self.actor.choose_action(state)

        return action

    def learn(self, state, reward, action, next_state):

        # gradient = grad[r + gamma * V(s_) - V(s)]
        td_error = self.critic.learn(state, reward, next_state)
        # true_gradient = grad[logPi(s,a) * td_error]
        self.actor.learn(state, action, td_error)

    def cal_reward(self, state, goal, reward):
        return reward

    def format_state(self, state):
        '''
        input:
            dict
        output:
            dict
        将state多余的feature筛除掉
        列表如下:
        [
            'pitch-deg', #飞机俯仰角
            'roll-deg' , #飞机滚转角
            'heading-deg' , #飞机朝向
            'vsi-fpm' ,  #爬升速度
            'uBody-fps', #飞机沿机身X轴的速度
            'vBody-fps', #飞机沿机身Y轴的速度
            'wBody-fps', #飞机沿机身Z轴的速度
            'x-accel-fps_sec', #飞机沿机身X轴的加速度
            'y-accel-fps_sec', #飞机沿机身Y轴的加速度
            'z-accel-fps_sec', #飞机沿机身z轴的加速度
        ]
        '''
        state_ = dict()
        # for feature in self.features:
        #     state_[feature] = state[feature]
        return state_

    def state2array(self, state):
        '''
        input:
            dict
        output:
            ndarray
        将state从dict形式转换为array
        
        '''
        state_ = np.array(list(state.values()))
        return state_

'''
class HLC():
    def __init__(self):
        pass

    def choose_action(self, state, goal):

        return action

    def learn(self, state, reward, action):
        pass

    def cal_reward(self, state):

        return reward

'''

class llcenv:
    def __init__(self, fgenv, features, action_bounds, goals):
        self.features = features
        self.action_bounds = action_bounds
        self.goal = goals
        self.fgenv = fgenv
        self.action_lowers,self.action_uppers = get_lower_upper_bound(self.action_bounds)
        self.action_space = gymspaces.Box(low =self.action_lowers , high=self.action_uppers)

    def step(self, action = np.array()):
        '''
        iuput: 
            action #actions in array type
        output:
            - ob
                true ob from fgenv + goal state
            - reward
                true reward from fgenv + llc reward
            - done
                whether done
        '''
        # self.fgenv = fgenv.fgenv()
        action_frame = '%f,%f,%f,%f,%f\n' % tuple(action.tolist())
        next_state, reward, done, _ = self.fgenv.step(action_frame)

        ob = self.format_state(next_state)

        ob_ = np.append(ob,self.state2array(self.goal))
        reward_ = self.cal_reward(next_state, self.goal , reward)

        return ob_, reward_ , done , _

    def cal_reward(self, state, goal, reward):
        error = 0.0
        for key in LLC_FEATURES:
            error += ((state[key]- goal[key])**2)
        error = error/len(LLC_FEATURES)
        
        return reward - 0.1*error

    def format_state(self, state, dtype = "array"):
        '''
        input:
            state(dict)
            dtype(str): #output dtype
                - array
                - dict
        output:
            dict
        将state多余的feature筛除掉
        列表如下:
        [
            'pitch-deg', #飞机俯仰角
            'roll-deg' , #飞机滚转角
            'heading-deg' , #飞机朝向
            'vsi-fpm' ,  #爬升速度
            'uBody-fps', #飞机沿机身X轴的速度
            'vBody-fps', #飞机沿机身Y轴的速度
            'wBody-fps', #飞机沿机身Z轴的速度
            'x-accel-fps_sec', #飞机沿机身X轴的加速度
            'y-accel-fps_sec', #飞机沿机身Y轴的加速度
            'z-accel-fps_sec', #飞机沿机身z轴的加速度
        ]
        '''
        state_ = dict()
        for feature in self.features:
            state_[feature] = state[feature]

        if dtype == "dict" :
            return state_
        if dtype == "array":
            return self.state2array(state_)

    def state2array(self, state):
        '''
        input:
            dict
        output:
            ndarray
        将state从dict形式转换为array
        
        '''
        state_ = np.array(list(state.values()))
        return state_
        

MAX_EPISODE = 1000
MAX_EP_STEPS = 200

if __name__ == "__main__":

    print("client begin!")
    # input("press enter to continue!")

    ## 初始化flightgear 通信端口
    fg2client_addr = ("127.0.0.1", 5700)
    client2fg_addr = ("127.0.0.1", 5701)
    telnet_addr = ("127.0.0.1", 5555)

    # 初始化flightgear 环境
    myfgenv = fgenv.fgenv(telnet_addr, fg2client_addr, client2fg_addr)
    initial_state = myfgenv.initial()

    myllc = llcenv(myfgenv, LLC_FEATURES, LLC_ACTION_BOUNDS,LLC_GOALS)

