# FG Autopilot

**important:** create folder`data/flylog` in the same directory before you run the code.

## environment

- python3.7
- pandas 

## code

- `autopilot.py` 主程序
    - 内部定义autopilot 类，包含pid自动驾驶的控制算法实现
    - 主程序为 `__main__`部分，详见注释
- `data_analysis.py`  用于分析飞行日志(`data/flylog`)，数据分析工具
- `DRLmodel` 存储深度强化学习模型代码，待实现
- `fgmodule` 存储我们编写的与`flightgear`通信的模块
    - `fgudp.py`  flightgear通信主要模块。状态接收和控制帧发送
    - `fgcmd.py`  实现fg远程命令行控制，复位等功能
- `modulesplus` 一些额外的模块
- code stucture  **`fgudp.py`已实现飞行日志保存** 

![struct](doc/struct_v0.1.4.jpg)

## data

- we will save all the fly log in folder `data/flylog`. their named by the time the log created.
- input format
    - input send from flight gear in every 0.1s
- **all the parameters are introduced in `data/fg参数_v0.pdf `**

## sys

- `takeoff ` 起飞
    - 起飞前
    - **跑道上** 加速与维持方向稳定
    - **离地后** 稳定爬升与维持机身姿态
- `cruise ` 巡航

## terms

- 水平飞行（horizontal flight）
- 定常飞行（steady flight）
- 操作杆转弯

## config & cmd

- `F:\FlightGear 2018.2.2\data\Docs\README.IO`

- setting for udp 
    ```
    --telnet=5555
    --httpd=5500
    --generic=socket,in,10,127.0.0.1,5701,udp,udp_input
    --generic=socket,out,10,127.0.0.1,5700,udp,udp_output
    ```

- [cmd line help doc link](http://flightgear.sourceforge.net/getstart-en/getstart-enpa2.html)

- [telnet help](http://wiki.flightgear.org/Telnet_usage#nasal?tdsourcetag=s_pctim_aiomsg) **this is very important for using telnet, please read carefully**

- quick replay
    Load Flight Recorder Tape (Shift-F1) load a pre-recorded flight to play back.
    Save Flight Recorder Tape (Shift-F2) saves the current flight to play back later.
    we can also use (Ctrl-R) to make a quick replay

- some important props 
    ```
    /sim/model/autostart  #设置飞机autostart的值
    /controls/gear/brake-parking #设置飞机停车刹车
    /sim/crashed #设置飞机是否坠毁
    ```

## issues
- pid control works not very well, 使用pid算法 飞行几分钟后，飞机会大幅度摇晃而失控坠机，估计为系统延时问题。
- fg飞机停止飞行但是未坠毁的bug

## TODO

#### fgenv

- 状态空间和动作空间的确定？

- 怎么判断飞行的阶段？以调用不同模型或reward函数？

- 怎么计算reward？，分阶段还是分模型
- 怎么判断飞行是否结束？（针对fg飞机停止飞行但是未坠毁的bug）
- 飞机状态数据处理？不同的量纲如何处理

#### DRL model

- 状态空间和动作空间的确定？
    -  如果强化学习算法给出结果足够快的话, 可以使用离散的$(-1,0,1)*delta$的动作空间？
- 训练过程保存，用保存的数据进行离线学习？
- 分阶段训练模型？

- DQN 
    - Q_value network 搭建
    - train method
- PPO2

## relative work

1， <https://blog.openai.com/openai-baselines-ppo/> 这个网站的第一个视频与咱们的工作有相似之处 OpenAI

2，<https://github.com/openai/baselines> OpenAI的深度强化学习库

3，<https://github.com/hill-a/stable-baselines>  一个基于上面的库的改进版本的库

4，<https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-1-A-DQN/> 莫凡python的DQN教程