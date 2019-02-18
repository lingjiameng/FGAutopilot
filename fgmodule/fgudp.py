import socket
import threading
import time
import datetime
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
            data_dict[data[0]] = float(data[1])
        except ValueError:
            if len(data[1]) == 0:
                data_dict[data[0]] = ""
            else:
                data_dict[data[0]] = parse_hms(data[1])
    return data_dict

def data2dict(data_frame):
    '''
    new version for data format
    Input:
        flightgear udp data frame
    Output:
        dict with all values are double
    v0.0 details:
        no clock time need to pay attention to.
        all the values are double
    '''
    data_list = data_frame.split(',')
    data_dict = dict()
    for data in data_list:
        data = data.split('=')
        data_dict[data[0]] = float(data[1])
    return data_dict

def dict2df(dict):
    '''
    trans dict to pandas DataFrame
    '''
    #关键点：将字典放入列表再转换可以避免错误
    return pd.DataFrame.from_dict([dict])

class fgudp:
    '''
    fgudp FlightGear udp communicate interface

    function:
    send control frame to flightgear
    recv state frame from flightgear
    '''
    def __init__(self, fg2client_addr, client2fg_addr, logpath="data/flylog"):
        self.fg2client_addr = fg2client_addr
        self.client2fg_addr = client2fg_addr
        self.logpath = logpath #飞行日志存储文件目录

        self.my_out_frames = Queue(100)
        self.inframe = dict()
        self.logfile = logpath + "log.csv"
        self.tostop = False
        self.t_alive = 0

    def stop(self):
        '''
        call this function to make all sub thread quit
        '''
        print("fgudp stop begin! waiting for 2 thread quit")
        self.tostop = True

        self.my_out_frames.put("") # make sender thread quit
        time.sleep(1)
        while self.t_alive:
            pass

    def start(self):
        if not os.path.exists(self.logpath):
            print(self.logpath, "not exists! create automatically!")
            os.makedirs(os.path.join(os.getcwd(), os.path.normpath(self.logpath))+"\\")
        else:
            print("flylog saving path check pass!")

        # initial some property of fgudp
        self.tostop = False
        self.inframe = dict()
        self.my_out_frames = Queue(100)
        self.logfile = self.logpath + "/log"+ gettime() + ".csv"

        t_rece = threading.Thread(
            target=self.receiver, args=(self.fg2client_addr,))
        t_sender = threading.Thread(
            target=self.sender, args=(self.client2fg_addr, self.my_out_frames))


        t_rece.daemon = True
        t_sender.daemon = True

        t_rece.start()
        t_sender.start()
        self.t_alive = 2

        self.block4ready()

    def receiver(self,fg2client_addr):
        rece = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rece.bind(fg2client_addr)

        # rece first frame as header
        data, addr = rece.recvfrom(1024)
        data = data.decode('utf-8')
        data_dict = data2dict(data)
        self.inframe = data_dict

        # intial log file and log head
        framebuffer = dict2df(data_dict)
        framebuffer.to_csv(self.logfile, mode='a', index=False, header=True)
        buffercount = 0

        # timebefore = time.clock()
        print('Bind UDP on %s:%s !' % fg2client_addr)
        print("save log to", self.logpath)
        while not self.tostop:
            # timenew =  time.clock()
            # print(timenew - timebefore)
            # timebefore = timenew
            
            data, addr = rece.recvfrom(1024)
            data = data.decode('utf-8')
            data_dict = data2dict(data)
            # test for delay
            # print("--------delay:{:.6f}------".format(time.time()-data_dict["time"]))
            # 0.000 or less than 0.0004
            self.inframe = data_dict

            framebuffer = framebuffer.append(
                dict2df(data_dict), ignore_index=True, sort=False)
            if(buffercount % 100 == 0):
                # print("save log to",self.logpath)
                # print(framebuffer)
                framebuffer.to_csv(self.logfile,mode = 'a', index=False,header=False)
                framebuffer = pd.DataFrame(data=None, columns=framebuffer.columns)
                buffercount = 0
            buffercount += 1

            # print('Received from %s:%s.' % addr, end=":")
            # print("recive frame", data_dict["frame"])
            # print("recive frame", data_dict["hi-heading"])
        #关闭udp通信接受端口
        print('UDP rece on %s:%s stop!' % fg2client_addr)
        rece.close()
        self.t_alive-=1

    def sender(self, client2fg_addr, out_frames=Queue()):
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        print("send data to %s:%s !" % client2fg_addr)

        count = 0
        while not self.tostop:
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

            # print('sender %d-th frame' % count)
        #关闭udp通信发送端口
        print("send data to %s:%s stop!" % client2fg_addr)
        sender.close()
        self.t_alive-=1

    def get_state(self):
        return copy.deepcopy(self.inframe)

    def send_controlframe(self, control_frame, delay = 0.0):
        #发送控制帧
        self.my_out_frames.put(control_frame)
        if delay:
            time.sleep(delay)
        #返回当前状态
        return copy.deepcopy(self.inframe)
    
    def block4ready(self):
        while not self.inframe:
            pass
        print("fgudp is ready!!!")


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
    # input("press enter to continue!")
    fg2client_addr = ("127.0.0.1", 5700)
    client2fg_addr = ("127.0.0.1", 5701)

    myfg = fgudp(fg2client_addr,client2fg_addr)

    for j in range(4):
        myfg.start()
        for i in range(10):
            # print(myfg.inframe)
            output = RL(myfg.inframe)
            myfg.send_controlframe(output)
        myfg.stop()

    #LAN
    # fg2client_addr = ("192.168.1.109", 5700)
    # client2fg_addr = ("192.168.1.101", 5701)
     
    # input()
    # print("[client shutdown after 100 seconds!]")
    # time.sleep(90)
    # print("[client shutdown after 10 seconds!]")
    # time.sleep(10)
    # print("[client shutdown!]")


if __name__ == "__main__":
    main()
