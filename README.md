# FG Autopilot

## environment

- python 3.6.7
- conda 4.5.12
- pandas 0.23.4
- numpy 1.14.2
- tensorflow 1.12.0
- [geographiclib](https://geographiclib.sourceforge.io/html/python/index.html) 1.49

## install

1. Install FlightGear in your computer

2. put `config/fgudp.xml`into folder `FG_ROOT/data/Protocol`

3. start flight gear and  add below command line option into flightgear setting page

    ```
    --allow-nasal-from-sockets
    --telnet=5555
    --httpd=5500
    --generic=socket,in,10,127.0.0.1,5701,udp,fgudp
    --generic=socket,out,10,127.0.0.1,5700,udp,fgudp
    ```

4. just type `python trainpilot.py`!! 

5. **[tips] next time you only need to start flightgear and do step 4. you don't need to do step1-3 again**

## code

- `trainpilot.py` 主程序
    - 训练模型主程序，已实现`pid`和简单`dqn`
- `data_analysis.py`  用于分析飞行日志(`data/flylog`)，数据分析工具
- `DRLmodel` 存储深度强化学习模型代码
    - `takeoff_dqn.py` 简单dqn模型用于起飞
    - PPO2 等模型待实现
- `fgmodule` 存储我们编写的与`flightgear`通信的模块
    - `fgudp.py`  flightgear通信主要模块。状态接收和控制帧发送
    - `fgcmd.py`  实现fg远程命令行控制，复位等功能
    - `fgenv.py` 将`fgudp` 和`fgcmd`进行封装, 详细功能见注释
        - `initial()`
        - `step()`
        - `reposition()` 
        - `reset()`
- `modulesplus` 一些额外的模块
    - `pidautopilot.py` 内部定义autopilot 类，包含pid自动驾驶的控制算法实现
- `doc` some important doc about flightgear and our research
    - **all the parameters are introduced in `doc/fg参数_v0.pdf `**
- `data` we will save all the fly log in folder `data/flylog`. their named by the time when the log created.
- code stucture  **`fgudp.py`已实现飞行日志保存** 

![struct](Doc/struct.jpg)

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
- **~~1）整合restart，简化训练形式。2）fgudp去除procer线程~~ 3)fgenv属性值初始化优化**
- ~~增加api，增加ob的数据结构，可以返回dict，以方便pid等其他算法使用~~

#### DRL model

- 状态空间和动作空间的确定？
    -  如果强化学习算法给出结果足够快的话, 可以使用离散的$(-1,0,1)*delta$的动作空间？
- 训练过程保存，用保存的数据进行离线学习？
- 分阶段训练模型？
- DQN 
    - Q_value network 搭建
    - train method
- Actor-Critic
- PPO2
- 结合pid和强化学习。对控制信息和状态转移进行融合，加入replay buffer 进行学习？ an important work to do

## relative work

1， <https://blog.openai.com/openai-baselines-ppo/> 这个网站的第一个视频与咱们的工作有相似之处 OpenAI

2，<https://github.com/openai/baselines> OpenAI的深度强化学习库

3，<https://github.com/hill-a/stable-baselines>  一个基于上面的库的改进版本的库

4，<https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-1-A-DQN/> 莫凡python的DQN教程