import matplotlib.pyplot as plt
import pandas as pd
import os
# pd.set_option('display.max_columns', None)
# # pd.set_option('display.max')
# logs = os.listdir("data/flylog/")

# df = pd.read_csv("data/flylog/log2019-1-16_21-42-23.csv")
# for log in logs:
#     if log[-3:] == "csv":
#         df_ = pd.read_csv("data/flylog/"+log)
#         df.append(df_)


# plt.subplot(211)
# plt.title("aileron")
# plt.plot(df.index, df.loc[:, ["aileron"]], color="b")

# plt.subplot(212)
# plt.title("elevator")
# plt.plot(df.index, df.loc[:, ["elevator"]], color="r")
# print(df.info())
# print(df.min())
# print(df.max())
# print(df.describe().loc[:, [
#     'aileron',  # 副翼 控制飞机翻滚 [-1,1]
#     'elevator',  # 升降舵 控制飞机爬升 [-1,1] 
#     'rudder',  # 方向舵 控制飞机转弯（地面飞机方向控制） [-1,1]
#     'throttle0',  # 油门0 [0,1]
#     'throttle1',  # 油门1 [0,1]
#     'flaps',  # 襟翼 在飞机起降过程中增加升力，阻力 全都是0
#     #TODO: 方向舵调整片
# ]])

# print(df.describe().loc[:, [
#     'pitch-deg',  # 飞机俯仰角 [-90,90]
#     'roll-deg',  # 飞机滚转角 [-180,180]
#     'heading-deg',  # 飞机朝向 [0, 360]
#     'vsi-fpm',  # 爬升速度 [0]
#     'uBody-fps',  # 飞机沿机身X轴的速度 [0,600]
#     'vBody-fps',  # 飞机沿机身Y轴的速度 [-200,200]
#     'wBody-fps',  # 飞机沿机身Z轴的速度 [-200, 200]
#     'x-accel-fps_sec',  # 飞机沿机身X轴的加速度 [0, 50]
#     'y-accel-fps_sec',  # 飞机沿机身Y轴的加速度 [-30, 30]
#     'z-accel-fps_sec',  # 飞机沿机身z轴的加速度 [-300,300]
# ]])


# plt.show()
def show_trajectory():
    df = pd.read_csv(
        "data/traindata/pid_traindata_sample.csv")
    plt.plot(df[ "latitude"], df["longitude"])
    plt.show()
    
show_trajectory()
