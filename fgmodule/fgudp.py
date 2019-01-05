import socket
import numpy as np
import threading
import time
import datetime
import pprint
import random
import os
import pandas as pd
from queue import Queue
import copy


RECEBUFFERSIZE = 100 #接收数据后存入文件的buffer大小
MAXHISTORYBUFFERSIZE = 20 #处理数据线程buffer存储历史数据上限


def parse_hms(s):
        hour_s, minute_s, second_s = s.split(':')
        return (int(hour_s), int(minute_s), int(second_s))

def gettime():
    now = datetime.datetime.now()
    date = "%s-%s-%s_%s-%s-%s" % (
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    return date

def format_data(data_frame):
    data_list = data_frame.split(',')
    data_dict = dict()
    for data in data_list:
        data = data.split('=')
        try:
            data_dict[data[0]] = [float(data[1])]
        except ValueError:
            if len(data[1]) == 0:
                data_dict[data[0]] = [""]
            else:
                data_dict[data[0]] = [parse_hms(data[1])]
    return data_dict

# def data2buffer(frames, historybuffer=[]):
#     while True:
#         historybuffer.insert(0, frames.get())

class fgudp():
    def __init__(self, fg2client_addr, client2fg_addr, logpath="data/flylog"):
        self.fg2client_addr = fg2client_addr
        self.client2fg_addr = client2fg_addr
        self.my_in_frames = Queue(100)
        self.my_out_frames = Queue(100)
        self.historybuffer = []
        self.inframe = dict()
        self.logpath = logpath #飞行日志存储文件目录



    def initalize(self):
        if not os.path.exists(self.logpath):
            print(self.logpath, "not exists! create automatically!")
            os.mkdir(self.logpath)
        else:
            print("flylog saving path check pass!")

        t_rece = threading.Thread(
            target=self.receiver, args=(self.fg2client_addr, self.my_in_frames))
        t_sender = threading.Thread(
            target=self.sender, args=(self.client2fg_addr, self.my_out_frames))
        t_procer = threading.Thread(
            target=self.procer, args=(self.my_in_frames, self.my_out_frames))

        t_rece.daemon = True
        t_procer.daemon = True
        t_sender.daemon = True

        t_rece.start()
        t_procer.start()
        t_sender.start()

        # 阻塞至飞机初始化完成
        self.inframe = self.my_in_frames.get()
        self.historybuffer.insert(0, self.inframe)
        pass

    def receiver(self,fg2client_addr, in_frames=Queue()):
        rece = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rece.bind(fg2client_addr)

        # rece first frame as header
        data, addr = rece.recvfrom(1024)
        data = data.decode('utf-8')
        data_dict = format_data(data)

        framebuffer = pd.DataFrame.from_dict(data_dict)
        
        logfile = self.logpath + "/log"+gettime() + ".csv"
        framebuffer.to_csv(logfile, mode='a', index=False, header=True)
        buffercount = 0

        # timebefore = time.clock()
        print('Bind UDP on %s:%s !' % fg2client_addr)
        while True:
            # timenew =  time.clock()
            # print(timenew - timebefore)
            # timebefore = timenew
            
            # 接收数据:
            ########################
            ## if out_frame not full, send one frame
            # else wait until it not empty
            # time.sleep(2)

            data, addr = rece.recvfrom(1024)
            data = data.decode('utf-8')
            data_dict = format_data(data)
            in_frames.put(data_dict) 

            framebuffer=framebuffer.append(pd.DataFrame.from_dict(data_dict),ignore_index=True,sort=False)
            if(buffercount % 100 == 0):
                print("save log to",self.logpath)
                # print(framebuffer)
                framebuffer.to_csv(logfile,mode = 'a', index=False,header=False)
                framebuffer = pd.DataFrame(data=None, columns=framebuffer.columns)
            buffercount += 1

            print('Received from %s:%s.' % addr, end=":")
            # print("recive frame", data_dict["frame"])
            print("recive frame", data_dict["hi-heading"])

    def sender(self, client2fg_addr, out_frames=Queue()):
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        print("send data to %s:%s !" % client2fg_addr)

        count = 0
        while True:
            # time.sleep(1)
            count += 1
            ########################
            ## if out_frame not empty, send one frame
            # else wait until it not empty
            # out_frames.not_empty.wait()
            control_frame = out_frames.get()
            # sender.sendto(b'sender %d-th frame' %
            #               count + control_frame.encode("utf-8"), client2fg_addr)
            sender.sendto(control_frame.encode("utf-8"), client2fg_addr)

            print('sender %d-th frame' % count)

    def procer(self, in_frames=Queue(), out_frames=Queue()):
        time.sleep(1)  # 等待接收线程初始化完成
        while True:
            #接受数据
            self.inframe = in_frames.get()
            timebefore = time.clock()
            self.historybuffer.insert(0, self.inframe)
            # 放止运算过快，而来不及得到状态信息就进行新的计算
            while not in_frames.empty():
                self.inframe = in_frames.get()
                self.historybuffer.insert(0, self.inframe)
                print("[historybuffer size : %d ]" % len(self.historybuffer))
            # delay about 0.005
            
            self.historybuffer = self.historybuffer[:MAXHISTORYBUFFERSIZE]
  
            print("[get input data once]")
            print(time.clock()-timebefore)
    
    def get_state(self):
        return copy.deepcopy(self.inframe),self.historybuffer

    def send_controlframe(self,control_frame):
        #发送控制帧
        self.my_out_frames.put(control_frame)
        #返回当前状态
        return copy.deepcopy(self.inframe),self.historybuffer


def RL(frame):
    '''[format as below inside [] ]control_frame with var_separator , 
    [%f, %f, %f, %f, %f\n]
    aileron, elevator, rudder, throttle0, throttle1
    副翼, 升降舵,方向舵, 油门0, 油门1
    '''

    control_frame = "0.0,,, 2.600000, 2.600000\n"
    time.sleep(0.7)
    return control_frame

def main():
    print("client begin!")
    input("press enter to continue!")
    fg2client_addr = ("127.0.0.1", 5700)
    client2fg_addr = ("127.0.0.1", 5701)

    myfg = fgudp(fg2client_addr,client2fg_addr)
    myfg.initalize()

    for i in range(100):
        print(myfg.inframe)
        output = RL(myfg.inframe)
        myfg.send_controlframe(output)

    #LAN
    # fg2client_addr = ("192.168.1.109", 5700)
    # client2fg_addr = ("192.168.1.101", 5701)
     
    input()
    # print("[client shutdown after 100 seconds!]")
    # time.sleep(90)
    # print("[client shutdown after 10 seconds!]")
    # time.sleep(10)
    # print("[client shutdown!]")


if __name__ == "__main__":
    main()
