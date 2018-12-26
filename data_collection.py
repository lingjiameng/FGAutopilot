import socket
import numpy
import threading
import time
import datetime
import pprint
import random
from queue import Queue
import pandas as pd


logDIR = "data/flylog"


def parse_hms(s):
    hour_s, minute_s, second_s = s.split(':')
    return [hour_s, minute_s, second_s]


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
                data_dict[data[0]] = ["-".join(parse_hms(data[1]))]
    return data_dict


def write_data():
    pass


def load_data():
    pass


def save_model():
    pass


def load_model():
    pass

# def data2buffer(frames, buffer=[]):
#     while True:
#         buffer.insert(0, frames.get())


def receiver(fg2client_addr, in_frames=Queue()):
    rece = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rece.bind(fg2client_addr)

    framebuffer = pd.DataFrame.from_dict(format_data("clock-indicated=0:0:0,aileron=0.0000,elevator=0.0000,rudder=0.0000,flaps=0.0000,throttle0=0.0000,throttle1=0.0000,vsi-fpm=0.0000,alt-ft=0.0000,ai-pitch=0.0000,ai-roll=0.0000,ai-offset=0.0000,hi-heading=0.0000,roll-deg=0.0000,pitch-deg=0.0000,heading-deg=0.0000,airspeed-kt=0.0000,speed-north-fps=0.0000,speed-east-fps=0.0000,speed-down-fps=0.0000,uBody-fps=0.0000,vBody-fps=0.0000,wBody-fps=0.0000,north-accel-fps_sec=0.0000,east-accel-fps_sec=0.0000,down-accel-fps_sec=0.0000,x-accel-fps_sec=0.0000,y-accel-fps_sec=0.0000,z-accel-fps_sec=0.0000,latitude=0.0000,longitude=0.0000,altitude=0.0000"))
    now = datetime.datetime.now()
    logname = logDIR+"/log%s-%s-%s_%s-%s-%s.csv" % (
        now.year, now.month, now.day, now.hour, now.minute, now.second)

    print('Bind UDP on %s:%s !' % fg2client_addr)

    framebuffer.to_csv(logname, mode='a', index=False, header=True)
    count = 0
    while True:
        # 接收数据:
        ########################
        ## if out_frame not full, send one frame
        # else wait until it not empty
        # time.sleep(2)

        data, addr = rece.recvfrom(1024)
        data = data.decode('utf-8')
        data_dict = format_data(data)
        framebuffer=framebuffer.append(pd.DataFrame.from_dict(data_dict),ignore_index=True,sort=False)

        # in_frames.put(data_dict)
        if(count % 100 == 0):
            print("save log")
            print(framebuffer)
            framebuffer.to_csv(logname,mode = 'a', index=False,header=False)
            framebuffer = pd.DataFrame(data=None, columns=framebuffer.columns)
        count+=1
        print('Received from %s:%s.' % addr, end=":")
        # print("recive frame", data_dict["frame"])
        print("recive frame", data_dict["clock-indicated"])


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
    buffer = []
    frame = dict()

    while True:
        #接受数据
        while not in_frames.empty():
            frame = in_frames.get()
            buffer.insert(0, frame)
            # print(frame)
            print("[buffer size : %d ]" % len(buffer))
            # pprint.pprint(frame)

        #强化学习
        print("[one time study!!!!!]")
        control_frame = RL(frame, buffer)

        #发送控制帧
        out_frames.put(control_frame)

    pass


def RL(frame, buffer):
    '''control_frame with var_separator , 
    %f, %f, %f, %f, %f
    aileron, elevator, rudder, throttle0, throttle1
    副翼, 电梯,方向舵, 油门0, 油门1
    '''

    control_frame = "0.0, 0.0, -2.00, 2.600000, 2.600000\n"
    time.sleep(random.randint(1, 3))
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
    # t_sender = threading.Thread(
    #     target=sender, args=(client2fg_addr, my_out_frames))
    # t_procer = threading.Thread(
    #     target=procer, args=(my_in_frames, my_out_frames))

    t_rece.daemon = True
    # t_procer.daemon = True
    # t_sender.daemon = True

    t_rece.start()
    # t_procer.start()
    # t_sender.start()
    print("[client shutdown after 1000 seconds!]")
    time.sleep(1000)
    print("[client shutdown after 10 seconds!]")
    time.sleep(10)
    print("[client shutdown!]")


if __name__ == "__main__":
    main()
