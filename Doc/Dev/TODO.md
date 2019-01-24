## TODO v0.2.0

#### fgenv

- ~~状态空间和动作空间的确定？~~
- ~~怎么判断飞行的阶段？以调用不同模型或reward函数？~~ 
- ~~怎么计算reward？，分阶段还是分模型~~ 以引导飞机稳定，实现控制器为目的设计reward
- ~~怎么判断飞行是否结束？（针对fg飞机停止飞行但是未坠毁的bug）~~ 不重要，使用复位至空中选项即可避免
- 飞机状态数据处理？不同的量纲如何处理。暂时直接利用变换范围归一化
- **~~1）整合restart，简化训练形式。2）fgudp去除procer线程~~ 3)fgenv属性值初始化优化**
- ~~增加api，增加ob的数据结构，可以返回dict，以方便pid等其他算法使用~~

#### DRL model

- 状态空间和动作空间的确定？
    - 如果强化学习算法给出结果足够快的话, 可以使用离散的$(-1,0,1)*delta$的动作空间？
- 训练过程保存，用保存的数据进行离线学习？
- 分阶段训练模型？
- ~~DQN~~ 效果不好，废弃 
- Actor-Critic
- PPO2
- 结合pid和强化学习。对控制信息和状态转移进行融合，加入replay buffer 进行学习？ an important work to do

### issues

- ~~pid control works not very well, 使用pid算法 飞行几分钟后，飞机会大幅度摇晃而失控坠机，估计为系统延时问题。~~ 是pid设计问题，已解决。
- FG飞机停止飞行但是未坠毁的bug。

### useful doc

1， <https://blog.openai.com/openai-baselines-ppo/> 这个网站的第一个视频与咱们的工作有相似之处 OpenAI

2，<https://github.com/openai/baselines> OpenAI的深度强化学习库

3，<https://github.com/hill-a/stable-baselines>  一个基于上面的库的改进版本的库

4，<https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-1-A-DQN/> 莫凡python的DQN教程



## TODO v0.3.0

