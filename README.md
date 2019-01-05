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
    - `fgcmd.py`  实现fg远程命令行控制，复位等功能（TODO）
- `modulesplus` 一些额外的模块
- code stucture

![struct](doc/struct_v0.1.1.jpg)

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
    --httpd=5500
    --generic=socket,in,10,127.0.0.1,5701,udp,udp_input
    --generic=socket,out,10,127.0.0.1,5700,udp,udp_output
    ```
- [cmd line help doc link](http://flightgear.sourceforge.net/getstart-en/getstart-enpa2.html)
- quick replay
    Load Flight Recorder Tape (Shift-F1) load a pre-recorded flight to play back.
    Save Flight Recorder Tape (Shift-F2) saves the current flight to play back later.

## issues
- pid control works not very well, 使用pid算法 飞行几分钟后，飞机会大幅度摇晃而失控坠机，估计为系统延时问题。