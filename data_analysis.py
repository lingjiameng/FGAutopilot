import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import os

LLC_FEATURE_BOUNDS = {
    'aileron': [-1, 1],  # 副翼 控制飞机翻滚 [-1,1] left/right
    'elevator': [-1, 1],  # 升降舵 控制飞机爬升 [-1,1] up/down
    'rudder': [-1, 1],  # 方向舵 控制飞机转弯（地面飞机方向控制） 0 /enter
    'throttle0': [0, 1],  # 油门0
    'throttle1': [0, 1],  # 油门1
    'pitch-deg': [-90., 90.],  # 飞机俯仰角
    'roll-deg': [-180., 180.],  # 飞机滚转角
    'heading-deg': [0., 360.],  # 飞机朝向
    'vsi-fpm': [0., 10.0],  # 爬升速度
    'uBody-fps': [0., 600.],  # 飞机沿机身X轴的速度
    'vBody-fps': [-200., 200.],  # 飞机沿机身Y轴的速度
    'wBody-fps': [-200., 200.],  # 飞机沿机身Z轴的速度
    'x-accel-fps_sec': [0., 50.],  # 飞机沿机身X轴的加速度
    'y-accel-fps_sec': [-30., 30.],  # 飞机沿机身Y轴的加速度
    'z-accel-fps_sec': [-300., 300.],  # 飞机沿机身z轴的加速度
}

DATA_BOUNDS = {
    'pitch-deg': [-90., 90.],  # 飞机俯仰角
    'roll-deg': [-180., 180.],  # 飞机滚转角
    'heading-deg': [0., 360.],  # 飞机朝向
    'vsi-fpm': [0., 10.0],  # 爬升速度
    'uBody-fps': [0., 600.],  # 飞机沿机身X轴的速度
    'vBody-fps': [-200., 200.],  # 飞机沿机身Y轴的速度
    'wBody-fps': [-200., 200.],  # 飞机沿机身Z轴的速度
    'x-accel-fps_sec': [0., 50.],  # 飞机沿机身X轴的加速度
    'y-accel-fps_sec': [-30., 30.],  # 飞机沿机身Y轴的加速度
    'z-accel-fps_sec': [-300., 300.],  # 飞机沿机身z轴的加速度
    'aileron': [-1, 1],  # 副翼 控制飞机翻滚 [-1,1] left/right
    'elevator': [-1, 1],  # 升降舵 控制飞机爬升 [-1,1] up/down
    'rudder': [-1, 1],  # 方向舵 控制飞机转弯（地面飞机方向控制） 0 /enter
    'throttle0': [0, 1],  # 油门0
    'throttle1': [0, 1],  # 油门1
    'flaps': [0, 1],  # 襟翼 在飞机起降过程中增加升力，阻力  Key[ / ]	Extend / retract flaps
    #TODO: 方向舵调整片
    'a_aileron': [-1, 1],  # 副翼 控制飞机翻滚 [-1,1] left/right
    'a_elevator': [-1, 1],  # 升降舵 控制飞机爬升 [-1,1] up/down
    'a_rudder': [-1, 1],  # 方向舵 控制飞机转弯（地面飞机方向控制） 0 /enter
    'a_throttle0': [0, 1],  # 油门0
    'a_throttle1': [0, 1],  # 油门1
}

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


def get_pid_state_trajectory():
    # altitude < 23 runway
    # altitude > 23 and < 6000 climbing
    # altitude > 6000 cruise
    mode = ["runway", "climbing", "cruise"]

    #get train data
    pid_traindata = pd.read_csv("data/traindata/pid_traindata_sample.csv")
    
    #获取飞行状态
    pid_traindata["fly_mode"] = (pid_traindata.loc[:, ["altitude"]] > 23).astype(
        int) + (pid_traindata.loc[:, ["altitude"]] > 6000).astype(int)

    groups = pid_traindata.groupby("fly_mode")
    for tag , group in groups:

        data_bounds = pd.DataFrame.from_dict(DATA_BOUNDS)

        norm_traindata = ((group - data_bounds.min()) /
                            (data_bounds.max() - data_bounds.min())).dropna(axis=1, how="any")
        
        #feed dict (state,target state), reward, new(state,target state)

        # #筛选部分作为targetstate
        # indexs =[i for i in range(0,norm_traindata.shape[0]-10,5)]
        # state = norm_traindata.loc[indexs, LLC_FEATURES]

        state = norm_traindata.loc[:, LLC_FEATURES]

        state.to_csv("data/traindata/trajectory_{}.csv".format(mode[tag]),index=False)


def get_raw_trajectory():
     # altitude < 23 runway
    # altitude > 23 and < 6000 climbing
    # altitude > 6000 cruise
    mode = ["runway", "climbing", "cruise"]

    #get train data
    pid_traindata = pd.read_csv("data/traindata/pid_traindata_sample.csv")

    #获取飞行状态
    pid_traindata["fly_mode"] = (pid_traindata.loc[:, ["altitude"]] > 23).astype(
        int) + (pid_traindata.loc[:, ["altitude"]] > 6000).astype(int)

    groups = pid_traindata.groupby("fly_mode")
    for tag, group in groups:

        state = group.loc[:, LLC_FEATURES]
        state_ =state.sample(frac = 0.2) 
        state_.to_csv(
            "data/traindata/trajectory_{}.csv".format(mode[tag]), index=False)

if __name__ == "__main__":
    # show_trajectory()
    # get_pid_state_trajectory()
    get_raw_trajectory()
