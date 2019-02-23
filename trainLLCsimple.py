
#%%
import time
import numpy as np


import LLC.LLCsimple as LLCsimple
import fgmodule.fgenv as fgenv

import scaffold.fgdata as dfer
import scaffold.pidpilot as PID
from scaffold.utils import gettime
##
epoch = 1000
step = 100

goals = {
    'pitch-deg': 0.,  # 飞机俯仰角
    'roll-deg': 0.,  # 飞机滚转角
    'heading-deg': 90.,  # 飞机朝向
}

#%%
myfgenv = fgenv.fgstart()

#%%
myllc = LLCsimple.LLC(n_old_actions=3)
# myllc.load("modelckpt/LLCsimple_DDPG2_delay/LLCsimple_DDPG.ckpt")

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
action = np.array([0., 0.])
old_action = np.array([0., 0.])

# #%%
# goal = goals
# action ,_ = myllc.choose_action(state,goal)
# print(LLCsimple.llc_reward(state , goal, old_action, action, 0))
# print(action)
# print(old_action)
# old_action = action


#%%
# import importlib
# importlib.reload(LLCsimple)


#%%
# 延时问题
# 模型输入输出的结构
# 训练trick
rewards = []
old_actions = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
next_old_actions = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
#%%
for e in range(epoch):
        state = myfgenv.replay("sky")
        time.sleep(2)
        
        goal = goals.copy()

        action = np.array([0., 0.])
        old_action = action
        old_actions = next_old_actions
        # goal_noise = np.random.randint(-30, 30)
        # goal['heading-deg'] = goal['heading-deg']+ goal_noise
        # intial_goal_diff = abs(goal['heading-deg']-state['heading-deg'])

        ep_reward = 0
        ep_diff = 0.0


        for s in range(step):
            # print("action and state ",old_action[0],state["aileron"])

            action, action_true = myllc.choose_action(state,goal,old_actions)
            
            elevator = -0.01*(0.0-state['pitch-deg'])
            diff = state['heading-deg']-goal['heading-deg']
            aileron = -0.1 * (float(state['roll-deg'])+diff)

            action_frame = dfer.action2frame((action_true[0],elevator,action_true[1],0.6,0.6))
            
            next_state, reward , done , info = myfgenv.step(action_frame,delay=0.8) 
            
            r_ = LLCsimple.llc_reward(state , goal,old_action,action ,reward)
            
            next_goal = goal
            next_old_actions = [action] + next_old_actions[:2]

            myllc.learn(state, goal, r_, action,next_state , next_goal,old_actions,next_old_actions)
            
            state = next_state
            goal = next_goal
            old_action = action
            old_actions = next_old_actions

            ep_diff += state['heading-deg'] - goal['heading-deg']
            ep_reward += r_
            if done:
                if s==0:
                    break
                print('Episode:', e, ' Reward: %.4f' %
                        (ep_reward/s),'diff: %.4f'% (ep_diff/s) ,'Explore: %.2f' % myllc.var, )

                rewards.append(ep_reward/s)
                break
            
            if s == step-1:
                print('Episode:', e, ' Reward: %.4f' %
                      (ep_reward/s), 'diff: %.4f' % (ep_diff/s), 'Explore: %.2f' % myllc.var, )
                # if ep_reward > -300:RENDER = True
                rewards.append(ep_reward/s)
                break


def pid():

    #%%
        #################################
        ###########PID
        # 延时问题
        # 模型输入输出的结构
        # 训练trick
    pidrewards = []

    for e in range(epoch):
        state = myfgenv.replay("sky")
        time.sleep(2)

        goal = goals.copy()

        # goal_noise = np.random.randint(-30, 30)
        # goal['heading-deg'] = goal['heading-deg']+ goal_noise
        # intial_goal_diff = abs(goal['heading-deg']-state['heading-deg'])

        ep_reward = 0
        ep_diff = 0.0

        for s in range(step):

            old_action = action

            elevator = -0.01*(0.0-state['pitch-deg'])
            diff = state['heading-deg']-goal['heading-deg']
            aileron = -0.1 * (float(state['roll-deg'])*0.15+0.3*diff)
            action = np.array([aileron, 0.0])

            action_frame = dfer.action2frame(
                (aileron, elevator, 0.0, 0.6, 0.6))

            next_state, reward, done, info = myfgenv.step(action_frame)

            r_ = LLCsimple.llc_reward(state, goal, old_action, action, reward)

            next_goal = goal

            state = next_state
            goal = next_goal
            ep_diff += state['heading-deg'] - goal['heading-deg']
            ep_reward += r_
            if done:
                if s == 0:
                    break
                print('Episode:', e, ' Reward: %i' %
                      (int(ep_reward)/s), 'diff: %.4f' % (ep_diff/s), 'Explore: %.2f' % myllc.var, )

                rewards.append(ep_reward/s)
                break

            if s == step-1:
                print('Episode:', e, ' Reward: %i' %
                      (int(ep_reward)/s), 'diff: %.4f' % (ep_diff/s), 'Explore: %.2f' % myllc.var, )
                # if ep_reward > -300:RENDER = True
                rewards.append(ep_reward/s)
                break


#%%
myllc.save("modelckpt/LLCsimple_DDPG2_delay/LLCsimple_DDPG.ckpt")

#%%
print(myllc.ddpg.pointer)
np.save("ddpgbuffer.npy",myllc.ddpg.memory)

#%%
print(rewards)


#%%
import matplotlib.pyplot as plt
plt.plot(rewards)
# plt.savefig("rewards.png")
plt.show()
