import time
import numpy as np
import DRL.DQN as tfdqn
import LLC.LLCsimple as LLCsimple
import LLC.LLC as LLC
import scaffold.fgdata as dfer
import scaffold.pidpilot as PID
import fgmodule.fgenv as fgenv
import pandas as pd
from scaffold.utils import gettime

'''
##########controlframe############


##########state_dict##############
{'aileron': 0.0, 'elevator': 0.0, 'rudder': -0.026, 'flaps': 0.0, 'throttle0': 0.6, 'throttle1': 0.6, 'vsi-fpm': 0.0, 'alt-ft': -372.808502, 'ai-pitch': 0.401045, 'ai-roll': 0.050598, 'ai-offset': 0.0, 'hi-heading': 80.568947, 'roll-deg': 0.050616, 'pitch-deg': 0.401055, 'heading-deg': 90.013458, 'airspeed-kt': 71.631187, 'speed-north-fps': -0.021637, 'speed-east-fps': 119.609383, 'speed-down-fps': -0.071344, 'uBody-fps': 119.606964, 'vBody-fps': -0.005778, 'wBody-fps': 0.765776, 'north-accel-fps_sec': 0.118887, 'east-accel-fps_sec': 5.870498, 'down-accel-fps_sec': -0.003219, 'x-accel-fps_sec': 6.095403, 'y-accel-fps_sec': -0.148636, 'z-accel-fps_sec': -32.113453, 'latitude': 21.325245, 'longitude': -157.93947, 'altitude': 22.297876, 'crashed': 0.0}
'''

actions_list = [
    'a_aileron',  # 副翼 控制飞机翻滚 [-1,1]
    'a_elevator',  # 升降舵 控制飞机爬升 [-1,1]
    'a_rudder',  # 方向舵 控制飞机转弯（地面飞机方向控制） [-1,1]
    'a_throttle0',  # 油门0 [0,1]
    'a_throttle1'  # 油门1 [0,1]
    # 'flaps',  # 襟翼 在飞机起降过程中增加升力，阻力 [0,1],实测影响不大，而且有速度限制
    #TODO: 方向舵调整片
]
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

LLC_FEATURE_BOUNDS = {
    'pitch-deg': [-90., 90.],  # 飞机俯仰角
    'roll-deg': [-180., 180.],  # 飞机滚转角
    'heading-deg': [0., 360.],  # 飞机朝向
    'vsi-fpm': [0., 10.0],  # 爬升速度
    'uBody-fps': [0., 600.],  # 飞机沿机身X轴的速度
    'vBody-fps': [-200., 200.],  # 飞机沿机身Y轴的速度
    'wBody-fps': [-200., 200.],  # 飞机沿机身Z轴的速度
    'x-accel-fps_sec': [0., 50.],  # 飞机沿机身X轴的加速度
    'y-accel-fps_sec': [-30., 30.],  # 飞机沿机身Y轴的加速度
    'z-accel-fps_sec': [-300., 300.],  # 飞机沿机身z轴的加速度
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
DATA_ACTIONS = [
    'a_aileron',  # 副翼 控制飞机翻滚 [-1,1]
    'a_elevator',  # 升降舵 控制飞机爬升 [-1,1]
    'a_rudder',  # 方向舵 控制飞机转弯（地面飞机方向控制） [-1,1]
    'a_throttle0',  # 油门0 [0,1]
    'a_throttle1'  # 油门1 [0,1]
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

DATA_BOUNDS = {
    'pitch-deg': [-90., 90.],  # 飞机俯仰角
    'roll-deg': [-180., 180.],  # 飞机滚转角
    'heading-deg': [0., 360.],  # 飞机朝向
    'vsi-fpm': [0., 10.0],  # 爬升速度
    'uBody-fps': [0., 600.],  # 飞机沿机身X轴的速度
    'vBody-fps': [-200., 200.],  # 飞机沿机身Y轴的速度
    'wBody-fps': [-200., 200.],  # 飞机沿机身Z轴的速度
    'x-accel-fps_sec': [0., 50.],  # 飞机沿机身X轴的加速度
    'y-accel-fps_sec': [-30., 30.],  # 飞机沿机身Y轴的加速度
    'z-accel-fps_sec': [-300., 300.],  # 飞机沿机身z轴的加速度
    'aileron': [-1, 1],  # 副翼 控制飞机翻滚 [-1,1] left/right
    'elevator': [-1, 1],  # 升降舵 控制飞机爬升 [-1,1] up/down
    'rudder': [-1, 1],  # 方向舵 控制飞机转弯（地面飞机方向控制） 0 /enter
    'throttle0': [0, 1],  # 油门0
    'throttle1': [0, 1],  # 油门1
    'flaps': [0, 1],  # 襟翼 在飞机起降过程中增加升力，阻力  Key[ / ]	Extend / retract flaps
    #TODO: 方向舵调整片
    'a_aileron': [-1, 1],  # 副翼 控制飞机翻滚 [-1,1] left/right
    'a_elevator': [-1, 1],  # 升降舵 控制飞机爬升 [-1,1] up/down
    'a_rudder': [-1, 1],  # 方向舵 控制飞机转弯（地面飞机方向控制） 0 /enter
    'a_throttle0': [0, 1],  # 油门0
    'a_throttle1': [0, 1],  # 油门1
}


MAX_EPISODE = 1000
MAX_EP_STEPS = 200

GAMMA = 0.9
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

##
epoch = 100
step = 300

train_data_dir = "data/traindata/"




def train_dqn():
    print("client begin!")
    # input("press enter to continue!")


    # 初始化flightgear 环境
    myfgenv = fgenv.fgstart()

    #初始化dqn模型
    mytfdqn = tfdqn.DQN(myfgenv.state_dim, 3)

    print("----------load model------------")
    mytfdqn.load('modelckpt/model.ckpt')
    ## 开始自动飞行
    for i in range(epoch):

        # reset flightgear

        state = myfgenv.reposition()
        time.sleep(2)
        for s in range(step):

            action = mytfdqn.egreedy_action(myfgenv.ob2array(state))

            # control frame
            # [ % f, % f, % f, % f, % f\n]
            # aileron, elevator, rudder, throttle0, throttle1
            action_frame = '%f,%f,%f,%f,%f\n' % (
                0.0, 0.0, float(state[2]+(0.01*(action-1))), 0.3, 0.3)

            next_state, reward, done, _ = myfgenv.step(action_frame)
            # if done:
            #     reward -=1000
            print(
                "-------------[action %d || reward %f]-----------" % (action, reward))
            mytfdqn.perceive(myfgenv.ob2array(state), action,
                             reward, myfgenv.ob2array(next_state), done)
            state = next_state
            # print(state)
            ##限制收发频率
            time.sleep(0.1)
            if done:
                break
        print("----------save model---------")
        mytfdqn.save('modelckpt/model.ckpt')
        if i % 10 == 9:
            myfgenv.reset()


'''
def test_llc_by_gym():
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(a_dim, s_dim, a_bound)

    myllc = LLC.LLC()

    var = 3  # control exploration
    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r / 10, s_)

            if ddpg.pointer > ddpg.MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                # if ep_reward > -300:RENDER = True
                break

    print('Running time: ', time.time() - t1)
'''

def train_llc():

    target_states = dfer.load_target_state("data/traindata")

    myfgenv = fgenv.fgstart()
    
    # bounds = {
    #     'rudder': [-1, 1],  # 方向舵 控制飞机转弯（地面飞机方向控制） 0 /enter
    # }

    myllc = LLC.LLC(LLC_FEATURE_BOUNDS,LLC_FEATURE_BOUNDS,LLC_ACTION_BOUNDS)

    for e in range(epoch):
        state = myfgenv.replay()
        time.sleep(2)

        goal_count = 0
        goal = dfer.get_target_state(state, target_states)

        ep_reward = 0

        for s in range(step):

            goal_count +=1

            action,action_true = myllc.choose_action(state,goal)

            action_frame = dfer.action2frame(action_true)
            next_state, reward , done , info = myfgenv.step(action_frame) 
            
            r_ = LLC.llc_reward(state , goal, reward)

            if goal_count%5 ==0:
                next_goal = dfer.get_target_state(next_state, target_states)

            myllc.learn(state, goal, r_, action,next_state , next_goal)
            
            state = next_state
            goal = next_goal

            ep_reward += r_
            if done:
                print('Episode:', e, ' Reward: %i' %
                      int(ep_reward), 'Explore: %.2f' % myllc.var, )
                break
            
            if s == step-1:
                print('Episode:', e, ' Reward: %i' %
                      int(ep_reward), 'Explore: %.2f' % myllc.var, )
                # if ep_reward > -300:RENDER = True
                break

########################################################
########### 自动飞行主程序 ###############################
if __name__ == "__main__":

    # pidfly()

    # train_dqn()

    # pid_datacol()

    train_llc()

