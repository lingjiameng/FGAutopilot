import time
import pprint
import numpy as np
import DRLmodel.takeoff_dqn as tfdqn
import fgmodule.fgudp as fgudp
import modulesplus.PID as PID
import fgmodule.fgcmd as fgcmd
import fgmodule.fgenv as fgenv
from autopilot import AutoPilot
from stable_baselines import PPO2
from stable_baselines import DQN


##
epoch = 10
step = 3000


########################################################
########### 自动飞行主程序 ###############################
if __name__ == "__main__":
    print("client begin!")
    # input("press enter to continue!")

    ## 初始化flightgear 通信端口
    fg2client_addr = ("127.0.0.1", 5700)
    client2fg_addr = ("127.0.0.1", 5701)
    telnet_addr = ("127.0.0.1", 5555)

    # myfg = fgudp.fgudp(fg2client_addr, client2fg_addr)
    # myfg.initalize()
    # myfgcmd = fgcmd.FG_CMD(telnet_addr)
    myfgenv = fgenv.fgenv(telnet_addr, fg2client_addr, client2fg_addr)
    initial_state =myfgenv.initial()

    ## 初始化自动驾驶模块
    # mypilot = AutoPilot(myfg.get_state()[0])
    mypilot = AutoPilot()
    

    # myfg.initalize()

    ## 开始自动飞行
    for i in range(epoch):
        # print(myfg.inframe)
        # print(myfg.get_state()[0]['frame'])
        ob = myfgenv.reset()
        mypilot.zero()
        for s in range(step):

            # just for test, so use not really ob
            ob, _ = myfgenv.fgudp.get_state()

            next_action = mypilot.pilot(ob)

            ob, reward, done, _ = myfgenv.step(next_action)
            if done:
                break

            print(ob)
            ##限制收发频率
            time.sleep(0.1)
