import math
from geographiclib.geodesic import Geodesic


def calreward(self, state_dict, state):
        '''
        print(state_dict)
        {'aileron': 0.0, 'elevator': 0.0, 'rudder': -0.026, 'flaps': 0.0, 'throttle0': 0.6, 'throttle1': 0.6, 'vsi-fpm': 0.0, 'alt-ft': -372.808502, 'ai-pitch': 0.401045, 'ai-roll': 0.050598, 'ai-offset': 0.0, 'hi-heading': 80.568947, 'roll-deg': 0.050616, 'pitch-deg': 0.401055, 'heading-deg': 90.013458, 'airspeed-kt': 71.631187, 'speed-north-fps': -0.021637, 'speed-east-fps': 119.609383, 'speed-down-fps': -0.071344, 'uBody-fps': 119.606964, 'vBody-fps': -0.005778, 'wBody-fps': 0.765776, 'north-accel-fps_sec': 0.118887, 'east-accel-fps_sec': 5.870498, 'down-accel-fps_sec': -0.003219, 'x-accel-fps_sec': 6.095403, 'y-accel-fps_sec': -0.148636, 'z-accel-fps_sec': -32.113453, 'latitude': 21.325245, 'longitude': -157.93947, 'altitude': 22.297876, 'crashed': 0.0}
        '''
        # 根据state_dict 自行判断飞机所在飞行模式,以按不同公式计算reward
        reward = 0.0

        # 需要用到的变量:
        # head_init : 每次初始化时 飞机的机头朝向 heading-deg
        # alti_init : 每次初始化时 飞机的海拔 altitude
        # vel_take_off : 起飞速度
        # vel_climb_max : 最大爬升速度（竖直向上），看老师提供的资料里应该是 50.5 fps
        # alti_low : 期望飞机所处的海拔范围的下界 ，参考老师给的资料，我设置的是 30000 feet
        # alti_high : 期望飞机所处的海拔范围的上界，我设置的是 31000 feet
        # vel_cruise : 平均巡航速度，与飞行的海拔有关，参考老师资料（与上面的预期海拔范围对应），我设置的是 679.85 fps
        # lati_tar: 目标点纬度 
        # long_tar: 目标点经度

        # 函数参数：
        # state: 状态变量:  0:跑道上，1:爬升阶段，2:巡航

        
        # 在跑道上 (任选一种方式)
        if state == 0:
        	delta_head = abs(state['heading-deg'] - head_init)
        	delta_head = min(delta_head, 360 - delta_head)

	        # 方式1: 考虑方向和速度
	        w_head_0 = 0.6	# 权重可调整
	        w_vel_0 = 0.4 	# 权重可调整
	        
	        reward = w_head_0 * math.exp(-1 * (delta_head**2)) + w_vel_0 * math.exp(-1 * (state_dict['uBody-fps'] - vel_take_off)**2)

	        # 方式2: 只考虑方向
	        reward = math.exp(-1 * (delta_head**2))


        # 爬升阶段
        elif state == 1:
        	vel_climb = -1 * state_dict['speed-down-fps']	# 爬升速度
        	altitude = state_dict['altitude']	# 飞机海拔 

        	if vel_climb <= vel_climb_max:		# 未超过最大爬升速度
				delta_head = abs(state['heading-deg'] - head_init)
        		delta_head = min(delta_head, 360 - delta_head)

        		w_head_1 = 0.4 	# 权重可调整
        		w_alti_1 = 0.6	# 权重可调整
        		
        		reward = w_head_1 * math.exp(-1 * (delta_head**2)) + w_alti_1 * math.exp(-1 * (altitude - alti_low)**2)

        	else:		# 超过最大爬升速度
        		reward = -1 * math.exp(-1 / (vel_climb - vel_climb_max)**2)


        # 巡航阶段
    	else:
    		altitude = state_dict['altitude']	# 飞机海拔
    		latitude = state_dict['latitude']	# 飞机纬度
    		longitude = state_dict['longitude']	# 飞机经度
    		heading = state_dict['heading-deg']	# 机头朝向
    		vel = state_dict['uBody-fps']		# 飞机沿机身x轴的速度
    		roll = state_dict['roll-deg']		# 飞机滚转角
    		pitch = state_dict['pitch-deg']		# 飞机俯仰角

    		w_alti_2 = 1	# 权重可调整
    		w_head_2 = 0.5	# 权重可调整
    		w_vel_2 = 0.3	# 权重可调整
    		w_roll = 0.1	# 权重可调整
    		w_pitch = 0.1	# 权重可调整

    		theta = Geodesic.WGS84.Inverse(latitude, longitude, lati_tar, long_tar)['azi1']	# 计算 target 相对于飞机的夹角
    		delta_head = heading - (180 - theta)
    		r_head_2 = 0.5 * math.cos(delta_head) + 0.5

    		r_alti_2 = 0.0				# 处于预期海拔范围内
    		if altitude < alti_low:		# 低于预期海拔范围下界
    			r_alti_2 = -1 * math.exp(-1 / (altitude - alti_low)**2)
    		elif altitude > alti_high:	# 高于预期海拔范围上界
    			r_alti_2 = -1 * math.exp(-1 / (altitude - alti_high)**2)   			

    		r_vel_2 = math.exp(-1 * (vel - vel_cruise)**2)

    		r_roll = math.exp(-1 * (abs(roll) / 10))
    		r_pitch = math.exp(-1 * (abs(pitch) / 10))

    		reward = w_head_2 * r_head_2 + w_alti_2 * r_alti_2 + w_vel_2 * r_vel_2 + w_roll * r_roll + w_pitch * r_pitch


        return reward

