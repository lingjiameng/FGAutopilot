import  scaffold.utils as utils

def pid(state, target_latitude=42, target_longitude=-122.4, target_altitude=7000):
    """
    给定状态，输出控制帧
    Inputs:
        state(dict): 飞行状态
    Returns:
        controls(tuple):控制量
            (aileron, elevator, rudder, throttle0, throttle1)
    """
    '''
            fly_mode(str):
            - "runway" 在跑道上
            - "climbing" 爬升阶段
            - "cruise" 巡航阶段
    control_frame(str):控制帧
    控制帧结构如下：
    control_frame with var_separator ,
    [%f, %f, %f, %f, %f \n]
    aileron, elevator, rudder, throttle0, throttle1
    '''
    fly_mode = ""

    aileron = state['aileron']
    elevator = state['elevator']
    rudder = state['rudder']
    throttle0 = state['throttle0']
    throttle1 = state['throttle1']

    tmp_mode = float(state["altitude"]) < target_altitude - 2000

    target_heading = utils.get_azimuth(
        state['longitude'], state['latitude'], target_longitude, target_latitude)
    heading_error = state['heading-deg'] - target_heading
    if heading_error > 180:
        heading_error -= 360
    if heading_error < -180:
        heading_error += 360

    if tmp_mode:  # 如果处于起飞模式
        # 如果在跑道上并且跑的还没到起飞速度
        if abs(float(state['speed-down-fps'])) < 1 and float(state['airspeed-kt']) < 120:
            fly_mode = "runway"
            if float(state['speed-north-fps']) < -0.0005:
                if float(state['north-accel-fps_sec']) < 0.0001:
                    rudder -= 0.001
            elif float(state['speed-north-fps'] > 0.0005):
                if float(state['north-accel-fps_sec']) > -0.0001:
                    rudder += 0.001
            if throttle0 < 0.6:
                throttle0 += 0.01
                throttle1 += 0.01

        else:  # 如果不在跑道上
            if float(state['speed-north-fps']) < -0.005:
                if float(state['north-accel-fps_sec']) < 0.01:
                    rudder -= 0.005
            elif float(state['speed-north-fps']) > 0.005:
                if float(state['north-accel-fps_sec']) > -0.01:
                    rudder += 0.005

        if float(state['speed-down-fps']) < -0.1 or float(state['airspeed-kt']) > 121:  # 起飞之后基本都能落到这里
            fly_mode = "climbing"
            if throttle0 < 0.6:  # 速度控制
                throttle0 += 0.01
                throttle1 += 0.01
            target_pitch_degree = (
                target_altitude-1000-float(state["altitude"])) * 0.01
            elevator = -0.004 * (target_pitch_degree - state['pitch-deg'])
            if float(state['roll-deg']) != 0:  # 翻滚控制
                aileron = -0.1 * float(state['roll-deg'])
            if aileron > 1:
                aileron = 1
            if aileron < -1:
                aileron = -1

    else:  # normal fly mode
        fly_mode = "cruise" #巡航阶段
        # PID计算转弯控制
        # heading_error/360.0 [-0.5,0.5]
        # kp = 0.1 
        # turn = kp * heading_error/360.0
        # print(" turn :",turn)
        # if float(state["roll-deg"]) != 0:
        #     aileron = -0.1 * float(state['roll-deg']) + turn
        if aileron > 1:
            aileron = 1
        if aileron < -1:
            aileron = -1
        target_pitch_degree = (target_altitude-float(state["altitude"]))*0.02
        target_pitch_degree = 0.0
        elevator = -0.005*(target_pitch_degree-state['pitch-deg'])

        rudder = 0
        throttle0 = 0.6
        throttle1 = 0.6

    # control = str(aileron)+","+str(elevator)+","+str(rudder) + \
    #     ","+str(throttle0)+","+str(throttle1)+"\n"  # type: str
    control = (aileron, elevator, rudder, throttle0, throttle1)
    return control
