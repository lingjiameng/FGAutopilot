
#%%
import time
import numpy as np


import LLC.LLCsimple as LLCsimple
import fgmodule.fgenv as fgenv

import scaffold.fgdata as dfer
import scaffold.pidpilot as PID
from scaffold.utils import gettime

#%%
LLC_FEATURE_BOUNDS = {
    'aileron': [-1, 1],  # 副翼 控制飞机翻滚 [-1,1] left/right
    'rudder': [-1, 1],  # 方向舵 控制飞机转弯（地面飞机方向控制） 0 /enter
    'pitch-deg': [-90., 90.],  # 飞机俯仰角
    'roll-deg': [-180., 180.],  # 飞机滚转角
    'heading-deg': [0., 360.],  # 飞机朝向
}
LLC_GOAL_BOUNDS = {
    'pitch-deg': [-90., 90.],  # 飞机俯仰角
    'roll-deg': [-180., 180.],  # 飞机滚转角
    'heading-deg': [0., 360.],  # 飞机朝向
}
goals = {
    'pitch-deg': 0.,  # 飞机俯仰角
    'roll-deg': 0.,  # 飞机滚转角
    'heading-deg': 90.,  # 飞机朝向
}

LLC_ACTION_BOUNDS = {
    'aileron': [-1, 1],  # 副翼 控制飞机翻滚 [-1,1] left/right
    'rudder': [-1, 1],  # 方向舵 控制飞机转弯（地面飞机方向控制） 0 /enter
}

##
epoch = 1000
step = 220


#%%
myfgenv = fgenv.fgstart()

# bounds = {
#     'rudder': [-1, 1],  # 方向舵 控制飞机转弯（地面飞机方向控制） 0 /enter
# }


#%%
myllc = LLCsimple.LLC(LLC_FEATURE_BOUNDS,LLC_GOAL_BOUNDS,LLC_ACTION_BOUNDS)


#%%
state = myfgenv.replay("sky")


#%%
state = {'aileron': 0.0,
 'elevator': 0.0,
 'rudder': 0.0,
 'flaps': 0.0,
 'throttle0': 0.0,
 'throttle1': 0.0,
 'vsi-fpm': 0.0,
 'alt-ft': -370.319122,
 'ai-pitch': 0.534232,
 'ai-roll': 0.277171,
 'ai-offset': 0.0,
 'hi-heading': 83.792358,
 'roll-deg': 0.056131,
 'pitch-deg': 0.467942,
 'heading-deg': 90.00074,
 'airspeed-kt': 1.532065,
 'speed-north-fps': -0.000281,
 'speed-east-fps': -0.007067,
 'speed-down-fps': 0.000105,
 'uBody-fps': -0.007067,
 'vBody-fps': 0.000281,
 'wBody-fps': 4.7e-05,
 'north-accel-fps_sec': -1.4e-05,
 'east-accel-fps_sec': -0.009769,
 'down-accel-fps_sec': -0.000142,
 'x-accel-fps_sec': 0.252822,
 'y-accel-fps_sec': -0.031483,
 'z-accel-fps_sec': -32.151367,
 'latitude': 21.325247,
 'longitude': -157.943137,
 'altitude': 21.358186,
 'crashed': 0.0}


#%%
goal = goals


#%%
myllc.goals

#%%
state

#%%
action = np.array([0., 0.])
old_action = np.array([0., 0.])

#%%
action ,_ = myllc.choose_action(state,goal)
print(LLCsimple.llc_reward(state , goal, old_action, action, 0))
print(action)
print(old_action)
old_action = action


#%%
import importlib
importlib.reload(LLCsimple)


#%%
for e in range(epoch):
       state = myfgenv.replay("sky")
       time.sleep(2)

       next_goal = goal

       ep_reward = 0

       for s in range(step):
       
           
           old_action = action 
           action,action_true = myllc.choose_action(state,goal)
           
           elevator = -0.01*(0.0-state['pitch-deg'])

           action_frame = dfer.action2frame((action_true[0],elevator,action_true[1],0.6,0.6))
           
           next_state, reward , done , info = myfgenv.step(action_frame) 
           
           r_ = LLCsimple.llc_reward(state , goal,old_action,action ,reward)
           
           next_goal = goals
           myllc.learn(state, goal, r_, action,next_state , next_goal)
           
           state = next_state
           goal = next_goal

           ep_reward += r_
           if done:
               print('Episode:', e, ' Reward: %i' %
                     int(ep_reward), 'Explore: %.2f' % myllc.var, )
               break
           
           if s == step-1:
               print('Episode:', e, ' Reward: %i' %
                     int(ep_reward), 'Explore: %.2f' % myllc.var, )
               # if ep_reward > -300:RENDER = True
               break
           time.sleep(0.1)


#%%
myllc.save("modelckpt/LLCsimple_DDPG/LLCsimple_DDPG.ckpt")
