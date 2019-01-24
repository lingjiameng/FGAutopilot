import socket
import time
import threading


def sender(out_addr):
    sender = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    count = 0
    while(True):
        time.sleep(1)
        count += 1
        data_frame = b"aileron=-0.900000,elevator=-1.000000,rudder=1.000000,flaps=0.375000,throttle0=1.000000,throttle1=1.000000,vsi-fpm=0.000000,alt-ft=30.058054,ai-pitch=-18.443966,ai-roll=-86.965988,ai-offset=0.000000,hi-heading=21.903837,roll-deg=-86.965991,pitch-deg=-18.443966,heading-deg=31.352836,airspeed-kt=120.111656,speed-north-fps=176.087646,speed-east-fps=83.970360,speed-down-fps=72.969231,uBody-fps=207.182892,vBody-fps=-8.866049,wBody-fps=-19.467810,north-accel-fps_sec=-43.516373,east-accel-fps_sec=2.719708,down-accel-fps_sec=-108.411034,x-accel-fps_sec=-78.381805,y-accel-fps_sec=123.183800,z-accel-fps_sec=18.470041,latitude=21.332679,longitude=-157.946107,altitude=31.399259,frame=%d,crashed=0,time=%f"%(count,time.time())
        sender.sendto(data_frame,out_addr)
        
        print("[frame %d have send!]"%count)

def receiver(in_addr):
    rece = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rece.bind(in_addr)
    count = 0

    while True:
        data_frame,addr = rece.recvfrom(1024)
        count+=1
        print("[frame %d have rece from %s:%s!:]"%((count,)+addr) , data_frame.decode("utf-8"))
    

if __name__ == "__main__":
    out_addr = ("127.0.0.1", 5700 )
    in_addr  = ("127.0.0.1", 5701 ) 

    print("[server begin run!]")
    t_sender=threading.Thread(target= sender,args=(out_addr,))
    t_rece  =threading.Thread(target= receiver, args=(in_addr,))

    t_sender.setDaemon(True)
    t_rece.setDaemon(True)
    t_sender.start()
    t_rece.start()
    
    print("[server shutdown after 100 seconds!]")
    time.sleep(90)
    print("[server shutdown after 10 seconds!]")
    time.sleep(10)
    print("[server shutdown!]")
