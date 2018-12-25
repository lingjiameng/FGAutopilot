import socket
import numpy
import threading
import time
import pprint
import random
from queue import Queue


def parse_hms(s):
    hour_s, minute_s, second_s = s.split(':')
    return (int(hour_s), int(minute_s), int(second_s))


def format_data(data_frame):
    data_list = data_frame.split(',')
    data_dict = dict()
    for data in data_list:
        data = data.split('=')
        try:
            data_dict[data[0]] = float(data[1])
        except ValueError:
            if len(data[1]) == 0:
                data_dict[data[0]] = None
            else:
                data_dict[data[0]] = parse_hms(data[1])
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
    print('Bind UDP on %s:%s !' % fg2client_addr)
    while True:
        # 接收数据:
        ########################
        ## if out_frame not full, send one frame
        # else wait until it not empty
        # time.sleep(2)

        data, addr = rece.recvfrom(1024)
        data = data.decode('utf-8')
        data_dict = format_data(data)
        in_frames.put(data_dict)

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
    fg2client_addr = ("192.168.1.109", 5700)
    client2fg_addr = ("192.168.1.101", 5701)

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

    print("[client shutdown after 100 seconds!]")
    time.sleep(90)
    print("[client shutdown after 10 seconds!]")
    time.sleep(10)
    print("[client shutdown!]")


if __name__ == "__main__":
    main()
