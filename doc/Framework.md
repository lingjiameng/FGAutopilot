## LLC

- 设计目标：
    - 尽量完成相应姿态控制 
    - 通过控制飞机相应设备完成一个姿态到另一个姿态的变换

- action：（* 表示主要控制量）

    1. 方向舵*
    2. 副翼*
    3. 升降舵*
    4. 油门*
    5. 襟翼 flaps
    6. 方向舵调整片

- state (针对飞机对象的状态参数，~~飞行员~~)

    ~~**所有姿态参数基于飞机坐标系**~~ 

    **所有的速度加速度矢量参数角度也是基于飞机坐标系**

    1. 姿态（模仿学习，match的时候要进行特殊处理，例如朝向应该考虑diff, 而不是从某一个值到另一个值）
        1. 飞机俯仰角（基于水平面）
        2. roll （飞机 滚转角 ）
        3. hi -heading仪表盘 显示的  飞机 机头 朝向 ，应该是 如图 以正北方向 为基准 ）
    2. 速度，加速度？

- goal （不可能超出state 和action space）

    - 基本等同于state
        - 按相同来做？ 难以调整到goal

- reward

    - 目前的想法
        - 模仿学习（利用state的 diff or distance 来计算）
    - 存在的问题
        - 只用state做评价，会有局限性。(+ action?). 
        - 根据正样本学习，会不有问题？
        - 连续动作， 由于连续性会产生不同reward？如何让reward有远见？
        - good motion ， diff or distance
        - **失败的情况，即飞机失控，和飞机坠机有很长一段时间** reward如何处理
        - etc.

# HLC

- 设计目标：
    - （在避开障碍物的情况下，尽量沿着航线飞）
    - 通过控制姿态，完成从一个空间点到另外一个空间点的变换

- action：（与LLC的state，goal 相同）
    1. idea1（LLC state 作为 action）
        1. 姿态调整
        2. ~~空间目标~~（因为这不属于LLC的范围）
- state（限定大小，只能感知到周围环境）
    1. LLC state 和LLC state 进行坐标系变换后的参数 *
    2. 飞机的 空间信息 *
    3. 风向
    4. 天气
    5. 航道信息 *？
    6. 地形和地图数据
- reward（在避开障碍物的情况下，尽量沿着航线飞）
    - 目前的想法
        - 根据天气和障碍物以及航道计算，例如：飞入雨云，给负的reward
        - 根据目标航线或者连续的目标点计算
    - 可能的 问题
        - 平稳性？
- goal  
    - 空间中的一点（经纬度，高度）
    - 航线（由连续的点构成）

## ~~HHLC~~

不需要学习

完全基于先验知识，规定航线中的点。

而HLC则是成空间变换到姿态序列的转换





## LLC frame

![Actor Critic](Framework/6-1-1-1547806816458.png)

### train

1. Actor

    1. transfer step:

        (state, target state) -> NN -> [[$\mu$, ... ,], [$\sigma $, ... ,]] -> N($\mu$,$\sigma^2$) -> actions

    2. backward:  

        ( td_error ) ->

2. Critic

    1. transfer step:

        (state, target state)  -> NN -> Q value

    2. backward:

        (reward, Q value) ->

3. how to choose target state for current state ?

4. how to design reward for Critic?

    1. separate in different part:
        1. $r_{safe}$ can trans to more parts like $r_{speed}$, $r_{angle}$ for safe
        2. $r_{dis2target}$ can trans to more parts like $r_{speed}$, $r_{heading}$for target
        3. etc.
    2. basic idea
        1. scale from 0 to 1
        2. different weight then sum together
        3. calculate base on (state, target state)

### pretrain

1. how to manager PID data?
    1. **how to find target state for now**
    2. how to design a temporal reward function for train ?
        1. all set to 1 ? because PID is a good pilot 

#### Critic

1.  train flow

    1. trans step:

        (state,target state), reward, new(state,target state) ->NN -> v_

    2. backward:

        (state,target state), reward -> ...  -> td_error 