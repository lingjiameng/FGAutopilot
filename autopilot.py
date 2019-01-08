import time
import DRLmodel.takeoff_dqn as tfdqn
import fgmodule.fgudp as fgudp
import modulesplus.PID as PID
import fgmodule.fgcmd as fgcmd

class AutoPilot():
    def __init__(self,frame):
        self.fly_mode = 1 #起飞模式  

        self.aileron = 0.0
        self.elevator = 0.0 
        self.rudder = 0.0 
        self.throttle0 = 0.0 
        self.throttle1 = 0.0 
        self.roll_1 = 0.0  # roll info of t-1
        self.roll_2 = 0.0  # roll info of t-2
    
    def zero(self):
        self.fly_mode = 1  # 起飞模式

        self.aileron = 0.0
        self.elevator = 0.0
        self.rudder = 0.0
        self.throttle0 = 0.0
        self.throttle1 = 0.0
        self.roll_1 = 0.0  # roll info of t-1
        self.roll_2 = 0.0  # roll info of t-2

    def frame2dict(self,frame):
        for key in frame.keys():
            frame[key] = frame[key][0]
        return frame


    def takeoff(self,frame, buffer):
        pass

    def pilot(self,frame,buffer):
        frame = self.frame2dict(frame)
        
        control_frame = self.pidcontrol(frame)
        # control_frame = ",,,,\n"
        return control_frame

    def save_model(self):
        pass

    def load_model(self):
        pass

    def pidcontrol(self,frame):

        if (self.fly_mode):  # 如果处于起飞模式

            # 如果在跑道上并且跑的还没到起飞速度
            if(abs(float(frame['speed-down-fps'])) < 1 and float(frame['airspeed-kt']) < 120):
                #print("on road")
                if(float(frame['speed-north-fps']) < -0.0005):
                    if(float(frame['north-accel-fps_sec']) < 0.0001):
                        self.rudder -= 0.0009
                elif(float(frame['speed-north-fps'] > 0.0005)):
                    if (float(frame['north-accel-fps_sec']) > -0.0001):
                        self.rudder += 0.001

                if(self.throttle0 < 0.6):
                    self.throttle0 += 0.01
                    self.throttle1 += 0.01

            else:  # 如果不在跑道上
                #print("in air")
                if(float(frame['speed-north-fps']) < -0.005):
                    if(float(frame['north-accel-fps_sec']) < 0.01):
                        self.rudder -= 0.005
                elif(float(frame['speed-north-fps']) > 0.005):
                    if(float(frame['speed-north-fps']) > -0.01):
                        self.rudder += 0.005
            if(float(frame['speed-down-fps']) < -1 or float(frame['airspeed-kt']) > 121):  # 起飞之后基本都能落到这里
                if (self.throttle0 < 0.6):  # 速度控制
                    self.throttle0 += 0.01
                    self.throttle1 += 0.01
                if(self.elevator > -0.1):  # 升降控制
                    self.elevator -= 0.001  # 如果上升速率比较慢，加大加快上升的速率
                elif(self.elevator <= -0.1 and self.elevator > -0.2):  # 如果上升速率比较快，减慢加快上升的速率
                    self.elevator -= 0.0001
                if(float(frame['roll-deg']) != 0):  # 翻滚控制
                    print(float(frame['roll-deg']), " ", self.roll_1, " ", self.roll_2)
                    # PI: 比例+微分(这里是差分)
                    self.aileron = -0.08 * \
                        float(frame['roll-deg'])-0.002 * \
                        (float(frame['roll-deg'])-self.roll_1)
                    self.roll_2 = self.roll_1
                    self.roll_1 = float(float(frame['roll-deg']))

            if(float(frame["altitude"]) > 3000):
                #print("high level")
                self.fly_mode = 0

            print(float(frame['speed-down-fps']))
            print("旋转 ", self.aileron, ' 升降 ', self.elevator, " 方向 ", self.rudder)
        else:  # normal fly mode
            print("high level")
            if(float(frame['altitude']) <= 4800):
                if(float(frame["roll-deg"]) != 0):
                    self.aileron = -0.08 * \
                        float(frame['roll-deg'])-0.2 * \
                        (float(frame['roll-deg'])-self.roll_1)
                    self.roll_1 = float(frame['roll-deg'])
                    self.roll_2 = self.roll_1
                    print(self.aileron, " ", self.roll_1, " ", self.roll_2)
                self.elevator = (float(frame["altitude"])-4333)*0.5
                self.rudder = 0
                self.throttle0 = 0.6
                self.throttle1 = 0.6
            print(float(frame['speed-down-fps']))
            print("旋转 ", self.aileron, ' 升降 ', self.elevator, " 方向 ", self.rudder)

        control_frame = "%f,%f,%f,%f,%f\n" % (self.aileron, self.elevator,
                                              self.rudder, self.throttle0, self.throttle1)
        return control_frame


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
    mypilot =  AutoPilot(myfg.get_state()[0])
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
            output = mypilot.pilot(myfg.get_state()[0],myfg.get_state()[1])
            
            myfg.send_controlframe(output)
            
            ##限制收发频率
            time.sleep(0.1)
            
        

        
        

