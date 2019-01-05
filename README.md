# FG Autopilot

**important:** create folder`data/flylog` in the same directory before you run the code.

## environment

- python3.7
- pandas 

## code

- `client.py` 主程序
    - 默认100s后关闭，如需更改，只需更改文件尾`time.sleep(90)`为`input()`,增加主程序休眠时间
    - 程序输入回车结束运行
- `server.py` 模仿flight gear 收发数据，用于代码调试
- `autopilot.py` 所有自动驾驶的控制算法实现的地方
- code stucture

![struct](doc/struct.jpg)

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

