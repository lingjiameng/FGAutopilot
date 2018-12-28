import socket
import numpy
import threading
import time
import datetime
import pprint
import random
import pandas as pd
from queue import Queue
import autopilot

logDIR = "data/flylog" #飞行日志存储文件目录
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


def write_data():
    pass


def load_data():
    pass


def save_model():
    pass


def load_model():
    pass

# def data2buffer(frames, historybuffer=[]):
#     while True:
#         historybuffer.insert(0, frames.get())


def receiver(fg2client_addr, in_frames=Queue()):
    rece = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rece.bind(fg2client_addr)

    # rece first frame as header
    data, addr = rece.recvfrom(1024)
    data = data.decode('utf-8')
    data_dict = format_data(data)

    framebuffer = pd.DataFrame.from_dict(data_dict)
    
    logfile = logDIR + "/log"+gettime() + ".csv"
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
            print("save log to",logDIR)
            # print(framebuffer)
            framebuffer.to_csv(logfile,mode = 'a', index=False,header=False)
            framebuffer = pd.DataFrame(data=None, columns=framebuffer.columns)
        buffercount += 1

        print('Received from %s:%s.' % addr, end=":")
        # print("recive frame", data_dict["frame"])
        print("recive frame", data_dict["hi-heading"])


def sender(client2fg_addr, out_frames=Queue()):
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


def procer(in_frames=Queue(), out_frames=Queue()):
    time.sleep(1) # 等待接收线程初始话完成
    historybuffer = []
    frame = dict()

    # frame = in_frames.get()
    # auto = autopilot.AutoPilot(frame["hi-heading"][0])

    while True:
        #接受数据
        # timebefore = time.clock()
        frame = in_frames.get()
        historybuffer.insert(0, frame)
        # 防止运算过快，而来不及得到状态信息就进行新的计算
        while not in_frames.empty():
            frame = in_frames.get()
            historybuffer.insert(0, frame)
            # print(frame)
            print("[historybuffer size : %d ]" % len(historybuffer))
            # pprint.pprint(frame)
        # print(time.clock()-timebefore)
        # delay about 0.005

        historybuffer = historybuffer[:MAXHISTORYBUFFERSIZE]
        ########### write your code below############
        ###input : frame and historybuffer
        #强化学习
        print("[one time study!!!!!]")
        # print(frame)
        # print(historybuffer)
        control_frame = RL(frame, historybuffer) 
        # control_frame = auto.takeoff(frame, historybuffer)


        #### output: control_frame
        #### format : [%f, %f, %f, %f, %f\n]
        #############################################

        #发送控制帧
        out_frames.put(control_frame)

    pass


def RL(frame, historybuffer):
    '''[format as below inside [] ]control_frame with var_separator , 
    [%f, %f, %f, %f, %f\n]
    aileron, elevator, rudder, throttle0, throttle1
    副翼, 升降舵,方向舵, 油门0, 油门1
    '''

    control_frame = "0.0,0.0,0.0, 2.600000, 2.600000\n"
    time.sleep(5)
    return control_frame


def main():
    print("client begin!")
    input("press enter to continue!")
    fg2client_addr = ("127.0.0.1", 5700)
    client2fg_addr = ("127.0.0.1", 5701)

    #LAN
    # fg2client_addr = ("192.168.1.109", 5700)
    # client2fg_addr = ("192.168.1.101", 5701)

    my_in_frames = Queue(100)
    my_out_frames = Queue(100)

    t_rece = threading.Thread(
        target=receiver, args=(fg2client_addr, my_in_frames))
    t_sender = threading.Thread(
        target=sender, args=(client2fg_addr, my_out_frames))
    t_procer = threading.Thread(
        target=procer, args=(my_in_frames, my_out_frames))

    t_rece.daemon = True
    t_procer.daemon = True
    t_sender.daemon = True

    t_rece.start()
    t_procer.start()
    t_sender.start()

    input()
    # print("[client shutdown after 100 seconds!]")
    # time.sleep(90)
    # print("[client shutdown after 10 seconds!]")
    # time.sleep(10)
    # print("[client shutdown!]")


if __name__ == "__main__":
    main()
