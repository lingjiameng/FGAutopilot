import time
import pprint
import numpy as np
import DRLmodel.takeoff_dqn as tfdqn
import modulesplus.pidautopilot as pidpilot

import fgmodule.fgenv as fgenv
from stable_baselines import PPO2
from stable_baselines import DQN
import os

'''
##########controlframe############


##########state_dict##############
{'aileron': 0.0, 'elevator': 0.0, 'rudder': -0.026, 'flaps': 0.0, 'throttle0': 0.6, 'throttle1': 0.6, 'vsi-fpm': 0.0, 'alt-ft': -372.808502, 'ai-pitch': 0.401045, 'ai-roll': 0.050598, 'ai-offset': 0.0, 'hi-heading': 80.568947, 'roll-deg': 0.050616, 'pitch-deg': 0.401055, 'heading-deg': 90.013458, 'airspeed-kt': 71.631187, 'speed-north-fps': -0.021637, 'speed-east-fps': 119.609383, 'speed-down-fps': -0.071344, 'uBody-fps': 119.606964, 'vBody-fps': -0.005778, 'wBody-fps': 0.765776, 'north-accel-fps_sec': 0.118887, 'east-accel-fps_sec': 5.870498, 'down-accel-fps_sec': -0.003219, 'x-accel-fps_sec': 6.095403, 'y-accel-fps_sec': -0.148636, 'z-accel-fps_sec': -32.113453, 'latitude': 21.325245, 'longitude': -157.93947, 'altitude': 22.297876, 'crashed': 0.0}
'''

##
epoch = 10000
step = 3000

########################################################
########### TODO: 自动飞行主程序 ###############################
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
        myfgenv.step("0.0,0.0,0.0,0.0,0.0\n")
        state = myfgenv.reset()
        myfgenv.step("0.0,0.0,0.0,0.0,0.0\n")
        time.sleep(2)
        for s in range(step):

            action = mytfdqn.egreedy_action(state)

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
            mytfdqn.perceive(state, action, reward, next_state, done)
            state = next_state
            # print(state)
            ##限制收发频率
            time.sleep(0.1)
            if done:
                break
        print("----------save model---------")
        mytfdqn.save('modelckpt/model.ckpt')
        if i % 10 == 9:
            myfgenv.stop()
            time.sleep(50)
            myfgenv = fgenv.fgenv(telnet_addr, fg2client_addr, client2fg_addr)
            initial_state = myfgenv.initial()
