from fgmodule.fgcmd import FG_CMD
from fgmodule.fgudp import fgudp
import time
import numpy as np

def fgstart(fg2client_addr=("127.0.0.1", 5700), client2fg_addr=("127.0.0.1", 5701), telnet_addr=("127.0.0.1", 5555)):
    '''
    简化FG通信启动过程
    ---
    Inputs:
        - fg2client_addr
        - client2fg_addr
        - telnet_addr
    Outputs:
        - myfgenv:初始话完成的fgenv类

    '''
    myfgenv = fgenv(telnet_addr, fg2client_addr, client2fg_addr)
    # initial_state = myfgenv.initial()
    myfgenv.initial()

    return myfgenv

class fgenv:
    '''
    python enviorment for flightgear

    more details in readme.md

    function:
        initial() need to be done right after class create
        step()
        reset()

    '''
    def __init__(self, telnet_addr, fg2client_addr, client2fg_addr, logpath="data/flylog"):
        self.fgcmd = FG_CMD(telnet_addr)
        self.fgudp = fgudp(fg2client_addr, client2fg_addr, logpath)

        self.action_space = None
        self.state_space = None
        self.state_dim = 0
        self.action_dim = 5
        self.initial_state = None #记录飞机初始状态 （数组）用于drl
        self.initial_state_dict = dict() # 同样记录飞机初始状态 （字典）用于reward设计

    def initial(self):
        # 先进行udp的初始化
        self.fgudp.start()
        self.fgcmd.start()
        self.fgcmd.auto_start()

        self.action_space = 5
        # control frame
        # [ % f, % f, % f, % f, % f\n]
        # aileron, elevator, rudder, throttle0, throttle1
        # 副翼, 升降舵, 方向舵, 油门0, 油门1

        self.initial_state = self.reset()
        self.state_space = len(self.initial_state.keys())
        self.state_dim = self.state_space
        return self.initial_state

    def step(self,action,delay = 0.1):
        """
            Parameters
            ----------
            action(str) :
                控制帧

            delay(double) : 
                delay定义多少延时之后返回状态帧

            Returns
            -------
            ob, reward, episode_over, info : tuple
                ob (dict) :
                    an environment-specific object representing your observation of
                    the environment.
                reward (float) :
                    amount of reward achieved by the previous action. The scale
                    varies between environments, but the goal is always to increase
                    your total reward.
                episode_over (bool) :
                    whether it's time to reset the environment again. Most (but not
                    all) tasks are divided up into well-defined episodes, and done
                    being True indicates the episode has terminated. (For example,
                    perhaps the pole tipped too far, or you lost your last life.)
                info (dict) :
                    diagnostic information useful for debugging. It can sometimes
                    be useful for learning (for example, it might contain the raw
                    probabilities behind the environment's last state change).
                    However, official evaluations of your agent are not allowed to
                    use this for learning.
        """
        # control frame
        # [ % f, % f, % f, % f, % f\n]
        # aileron, elevator, rudder, throttle0, throttle1
        # 副翼, 升降舵, 方向舵, 油门0, 油门1

        # TODO: 暂时假设输入的action为control_frame的格式
        # 估计以后要修改为多种形式
        # dqn 为离散型的值，然后对应五个控制量，需要转换 并且格式化为control frame
        # ppo2 为连续的值，可能只需要格式化
        state_dict = self.fgudp.send_controlframe(action,delay)
        
        # TODO: 增加数据的归一化过程 效果会更好
        ob = self.state_dict2ob(state_dict)

        # reward 需要增加很多东西
        reward = self.calreward(state_dict)

        # 暂时已完成，用flightgear内部的crashed变量实现
        episode_over = self.judge_over(state_dict)
        
        info = None

        return  ob, reward, episode_over, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        but due to the long time for flightgear reset, use function **reposition()** as more as possible 

        Returns: observation (object): the initial observation of the
            space.
        """
        #关闭udp通信端口
        self.fgudp.stop()

        ## 发送reset指令
        self.fgcmd.reset()
        time.sleep(4) # 等待四秒以确认发出
        # 关闭 telent 通信端口
        self.fgcmd.stop()

        #fg 通信端口全部关闭，等待重启
        time.sleep(50)

        # flight gear重启完成后，重启通信端口
        self.fgudp.start()
        self.fgcmd.start()

        # 飞机autostart
        self.fgcmd.auto_start()

        state_dict = self.fgudp.get_state()

        # TODO: 临时初始化飞机初始状态
        self.initial_state_dict = state_dict

        ob = self.state_dict2ob(state_dict)

        return ob


    def reposition(self):
        """
        In this function, we only reposition the plane due to the fact that reset while take a long time. 
        Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        '''
        使用三次空白控制帧以复位飞机控制器
        '''
        self.fgudp.send_controlframe("0.0,0.0,0.0,0.0,0.0\n")
        # 飞机reposition指令
        self.fgcmd.reposition()
        self.fgudp.send_controlframe("0.0,0.0,0.0,0.0,0.0\n")
        # 等待4秒以提供flight gear运算
        time.sleep(3)
        self.fgudp.send_controlframe("0.0,0.0,0.0,0.0,0.0\n")

        # 获取flight gear 初始状态
        state_dict = self.fgudp.get_state()
        self.initial_state_dict = state_dict
        ob = self.state_dict2ob(state_dict)
        return ob

    def replay(self, pos = "ground"):
        """
        In this function, we only reposition the plane use fg cmd replay 
        Resets the state of the environment and returns an initial observation.
        Inputs:
            pos(str) # the position you want go to
                -"ground" default
                -"sky" 
        Returns: observation (object): the initial observation of the
            space.
        """
        '''
        使用三次空白控制帧以复位飞机控制器
        '''
        self.fgudp.send_controlframe("0.0,0.0,0.0,0.0,0.0\n")
        # 飞机replay指令
        self.fgcmd.replay(pos)
        # self.fgudp.send_controlframe("0.0,0.0,0.0,0.0,0.0\n")
        # 等待4秒以提供flight gear运算
        time.sleep(3)
        self.fgcmd.auto_start()
        # self.fgudp.send_controlframe("0.0,0.0,0.0,0.0,0.0\n")

        # 获取flight gear 初始状态
        state_dict = self.fgudp.get_state()
        self.initial_state_dict = state_dict
        ob = self.state_dict2ob(state_dict)
        return ob

    def render(self, mode='human'):

        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        pass

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        print("Could not seed environment %s", self)
        return

    def state_dict2ob(self, state_dict):

        #增加数据的归一化过程 效果会更好
        # ob = np.array(list(state_dict.values()))
        ob = state_dict
        return ob

    def ob2array(self, ob):
        _ob = np.array(list(ob.values()))
        return _ob

    def calreward(self, state_dict):
        '''
        print(state_dict)
        {'aileron': 0.0, 'elevator': 0.0, 'rudder': -0.026, 'flaps': 0.0, 'throttle0': 0.6, 'throttle1': 0.6, 'vsi-fpm': 0.0, 'alt-ft': -372.808502, 'ai-pitch': 0.401045, 'ai-roll': 0.050598, 'ai-offset': 0.0, 'hi-heading': 80.568947, 'roll-deg': 0.050616, 'pitch-deg': 0.401055, 'heading-deg': 90.013458, 'airspeed-kt': 71.631187, 'speed-north-fps': -0.021637, 'speed-east-fps': 119.609383, 'speed-down-fps': -0.071344, 'uBody-fps': 119.606964, 'vBody-fps': -0.005778, 'wBody-fps': 0.765776, 'north-accel-fps_sec': 0.118887, 'east-accel-fps_sec': 5.870498, 'down-accel-fps_sec': -0.003219, 'x-accel-fps_sec': 6.095403, 'y-accel-fps_sec': -0.148636, 'z-accel-fps_sec': -32.113453, 'latitude': 21.325245, 'longitude': -157.93947, 'altitude': 22.297876, 'crashed': 0.0}
        '''
        # 根据state_dict 自行判断飞机所在飞行模式,以按不同公式计算reward
        reward = 0.0
        head_init = self.initial_state_dict['heading-deg']
        # head_init是每次初始化时飞机的heading-deg
        if state_dict["altitude"]<23 :
            reward = np.exp(-1 * (min(abs(state_dict['heading-deg'] - head_init), 360 - abs(
                state_dict['heading-deg'] - head_init)))**2)
            reward -= 1.0
        return reward

    def judge_over(self, state_dict):
        # 判断此次飞行模拟是否结束，即flighgear中飞机是否坠毁
        over = False

        count = 0 
        # crash
        if int(state_dict["crashed"])==1:
            over = True
        # # stable
        # if abs(state_dict["airspeed-kt"]) < 1.3:
        #     count+=1
        # else:
        #     count = 0
        # if(count >100):
        #     over = True

        # out runway 21.325247
        # if abs(state_dict["latitude"]-21.325247) > 0.0004:
        #     over = True

        return over
    
    def stop(self):
        '''
        
        '''
        # 重置flightgear环境
        self.fgcmd.reset()
        time.sleep(2)
        # 关闭fgudp 和fgcmd 端口 
        self.fgudp.stop()
        self.fgcmd.stop()

    
