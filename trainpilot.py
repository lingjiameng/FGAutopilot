import time
import pprint
import numpy as np
import DRLmodel.takeoff_dqn as tfdqn
import scaffold.pidpilot as PID
import datetime
import fgmodule.fgenv as fgenv
import pandas as pd
import os

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


def gettime():
    """
    return a time str for now
    """
    now = datetime.datetime.now()
    date = "%s-%s-%s_%s-%s-%s" % (
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    return date
##
epoch = 10000
step = 3000

train_data_dir = "data/traindata/"

## pid control example
def pid_datacol():
    print("client begin!")
    # input("press enter to continue!")

    ## 初始化flightgear 通信端口
    fg2client_addr = ("127.0.0.1", 5700)
    client2fg_addr = ("127.0.0.1", 5701)
    telnet_addr = ("127.0.0.1", 5555)

    myfgenv = fgenv.fgenv(telnet_addr, fg2client_addr, client2fg_addr)
    initial_state =myfgenv.initial()

    train_data_file = train_data_dir+"traindata"+gettime()+".csv"

    ## 开始自动飞行
    for i in range(epoch):
        # print(myfg.inframe)
        # print(myfg.get_state()[0]['frame'])
        state = myfgenv.replay()
        time.sleep(2)

        # 获取训练数据文件名和文件头
        train_data_file = train_data_dir+"traindata"+gettime()+".csv"
        train_data_cols = list(state.keys()) + actions_list

        # 生成文件头
        framebuffer = pd.DataFrame(data=None, columns=train_data_cols)
        framebuffer.to_csv(train_data_file, mode='a',
                           index=False, header=True)
        buffercount = 0

        for s in range(10*step):

            # just for test, so use not really ob

            action = PID.pid(state)
            action_frame = "%f,%f,%f,%f,%f\n" % action
            next_state , reward , done, _ = myfgenv.step(action_frame)
            
            #####
            # 存储内容
            # state, action, reward(暂时不需要保存), done(包含在state的crashed中)
            # 
            # state(dict)

            framebuffer.loc[buffercount] = list(state.values())+list(action)
            if(buffercount % 100 == 0):
                # print("save log to",self.logpath)
                # print(framebuffer)
                framebuffer.to_csv(train_data_file, mode='a',
                                   index=False, header=False)
                framebuffer = pd.DataFrame(
                    data=None, columns=train_data_cols)
                buffercount = 0
            buffercount += 1

            if done:
                break
            state = next_state
            ##限制收发频率
            time.sleep(0.1)

def train_dqn():
    print("client begin!")
    # input("press enter to continue!")

    ## 初始化flightgear 通信端口
    fg2client_addr = ("127.0.0.1", 5700)
    client2fg_addr = ("127.0.0.1", 5701)
    telnet_addr = ("127.0.0.1", 5555)

    # 初始化flightgear 环境
    myfgenv = fgenv.fgenv(telnet_addr, fg2client_addr, client2fg_addr)
    initial_state = myfgenv.initial()

    #假设
    print(initial_state)
    print("state dim", myfgenv.state_dim)
    #初始化dqn模型
    mytfdqn = tfdqn.DQN(myfgenv.state_dim, 3)

    # if os.path.exists('modelckpt/model.ckpt'):
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

def collect_data():
    print("client begin!")
    # input("press enter to continue!")

    ## 初始化flightgear 通信端口
    fg2client_addr = ("127.0.0.1", 5700)
    client2fg_addr = ("127.0.0.1", 5701)
    telnet_addr = ("127.0.0.1", 5555)

    # 初始化flightgear 环境
    myfgenv = fgenv.fgenv(telnet_addr, fg2client_addr, client2fg_addr)
    myfgenv.initial()
    input("enter to quit")




########################################################
########### 自动飞行主程序 ###############################
if __name__ == "__main__":

    # pidfly()

    # train_dqn()

    pid_datacol()
