import time
import numpy as np
import scaffold.pidpilot as PID

import fgmodule.fgenv as fgenv
import pandas as pd

from scaffold.utils import gettime


actions_list = [
    'a_aileron',  # 副翼 控制飞机翻滚 [-1,1]
    'a_elevator',  # 升降舵 控制飞机爬升 [-1,1]
    'a_rudder',  # 方向舵 控制飞机转弯（地面飞机方向控制） [-1,1]
    'a_throttle0',  # 油门0 [0,1]
    'a_throttle1'  # 油门1 [0,1]
    # 'flaps',  # 襟翼 在飞机起降过程中增加升力，阻力 [0,1],实测影响不大，而且有速度限制
    #TODO: 方向舵调整片
]

##
epoch = 10000
step = 3000

train_data_dir = "data/traindata/"


## pid control example
def pid_datacol():
    print("client begin!")
    # input("press enter to continue!")

    ## 初始化flightgear 通信端口
    myfgenv = fgenv.fgstart()

    train_data_file = train_data_dir+"traindata"+gettime()+".csv"

    ## 开始自动飞行
    for i in range(epoch):
        # print(myfg.inframe)
        # print(myfg.get_state()[0]['frame'])
        state = myfgenv.replay()
        time.sleep(2)

        # 获取训练数据文件名和文件头
        train_data_file = train_data_dir+"traindata"+gettime()+".csv"
        train_data_cols = list(state.keys()) + actions_list + ["fly_mode"]

        # 生成文件头
        framebuffer = pd.DataFrame(data=None, columns=train_data_cols)
        framebuffer.to_csv(train_data_file, mode='a',
                           index=False, header=True)
        buffercount = 0

        for s in range(10*step):

            # just for test, so use not really ob
            fly_mode = int(state["altitude"] > 23) + \
                int(state["altitude"] > 6000)

            action= PID.pid(state)
            action_frame = "%f,%f,%f,%f,%f\n" % action
            next_state, reward, done, _ = myfgenv.step(action_frame)

            #####
            # 存储内容
            # state, action, reward(暂时不需要保存), done(包含在state的crashed中)
            #
            # state(dict)

            framebuffer.loc[buffercount] = list(
                state.values())+list(action) + [fly_mode]
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





# TODO: 手动飞行数据收集函数
def datacol():
    print("client begin!")
    # input("press enter to continue!")

    ## 初始化flightgear 通信端口
    fg2client_addr = ("127.0.0.1", 5700)
    client2fg_addr = ("127.0.0.1", 5701)
    telnet_addr = ("127.0.0.1", 5555)

    myfgenv = fgenv.fgenv(telnet_addr, fg2client_addr, client2fg_addr)
    initial_state = myfgenv.initial()

    train_data_file = train_data_dir+"traindata"+gettime()+".csv"

    ## 开始自动飞行
    for i in range(epoch):
        # print(myfg.inframe)
        # print(myfg.get_state()[0]['frame'])
        state = myfgenv.replay()
        time.sleep(2)

        # 获取训练数据文件名和文件头
        train_data_file = train_data_dir+"traindata"+gettime()+".csv"
        train_data_cols = list(state.keys()) + actions_list + ["fly_mode"]

        # 生成文件头
        framebuffer = pd.DataFrame(data=None, columns=train_data_cols)
        framebuffer.to_csv(train_data_file, mode='a',
                           index=False, header=True)
        buffercount = 0

        for s in range(10*step):

            # just for test, so use not really ob
            fly_mode = int(state["altitude"] > 23) + \
                int(state["altitude"] > 6000)
            action = PID.pid(state)
            action_frame = "%f,%f,%f,%f,%f\n" % action
            next_state, reward, done, _ = myfgenv.step(action_frame)

            #####
            # 存储内容
            # state, action, reward(暂时不需要保存), done(包含在state的crashed中)
            #
            # state(dict)

            framebuffer.loc[buffercount] = list(
                state.values())+list(action) + [fly_mode]
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
