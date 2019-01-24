import time
import scaffold.fgdata as dfer  # 数据处理模块
import scaffold.pidpilot as PID   # PID 自动驾驶程序
import fgmodule.fgenv as fgenv  # flight gear 通信模块

print("client begin!")
# input("press enter to continue!")

## 初始化flightgear通信端口
myfgenv = fgenv.fgstart()


epoch = 1000

## 开始自动飞行
for i in range(epoch):

    state = myfgenv.replay() ## 复位飞机至飞行起点

    while True:

        action = PID.pid(state) #调用PID模块根据状态生成控制量

        # 格式化控制量为规定格式的控制帧
        action_frame = "%f,%f,%f,%f,%f\n" % action

        # FG执行控制帧给定的相应动作，返回下一个状态的观察值
        next_state, reward, done, _ = myfgenv.step(action_frame)

        if done:# 如果坠机，结束循环
            break

        state = next_state #更新状态为观察到的新的状态

        ##人工加入延时,限制收发频率。此处并不好，待修改
        time.sleep(0.1)
