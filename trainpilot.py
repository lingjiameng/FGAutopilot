import time
import pprint
import numpy as np
import DRLmodel.takeoff_dqn as tfdqn
import fgmodule.fgudp as fgudp
import modulesplus.PID as PID
import fgmodule.fgcmd as fgcmd
from autopilot import AutoPilot
from stable_baselines import PPO2
from stable_baselines import DQN


##
epoch = 10
step = 1000


########################################################
########### 自动飞行主程序 ###############################
if __name__ == "__main__":
    print("client begin!")
    # input("press enter to continue!")

    ## 初始化flightgear 通信端口
    fg2client_addr = ("127.0.0.1", 5700)
    client2fg_addr = ("127.0.0.1", 5701)
    telnet_addr = ("127.0.0.1", 5555)

    myfg = fgudp.fgudp(fg2client_addr, client2fg_addr)
    myfg.initalize()
    myfgcmd = fgcmd.FG_CMD(telnet_addr)

    ## 初始化自动驾驶模块
    mypilot = AutoPilot(myfg.get_state()[0])
    myfgcmd.auto_start()

    ## 开始自动飞行
    for i in range(epoch):
        # print(myfg.inframe)
        # print(myfg.get_state()[0]['frame'])
        myfgcmd.reposition()
        mypilot.zero()
        time.sleep(5)
        mypilot.zero()
        for s in range(step):
            state_dict = myfg.get_state()[0]
            state = np.array([value for value in state_dict.values()])
            # print(state_dict,)
            # print(state)
            # break
            output = mypilot.pilot(myfg.get_state()[0], myfg.get_state()[1])

            myfg.send_controlframe(output)

            ##限制收发频率
            time.sleep(0.1)

        # break
