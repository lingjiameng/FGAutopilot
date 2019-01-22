import numpy as np
import pandas as pd


def bounds_dict2arr(bounds):
    """
    转换bounds为 numpy(2,N)维数组
    ----
    Inputs:
        bounds(dict): 
            {"feature": [lower, upper],...}
    Returns:
        bound_numpy_array: 顺序按照字典中key的顺序不变
            (2,n) dtype = np.float (float32)
            [
                [lower1,l2,l3,...],
                [upper1,u2,u3,...]
            ]
    """
    bounds_tmp = np.array(list(bounds.values())).astype(np.float)
    bounds_tmp = bounds_tmp.T
    return bounds_tmp


def filter_state(state,features = [],bounds=dict(),objtype = "dict"):
    '''
    过滤输入状态(state)多余的特征(feature)
    ---
    Inputs:
        - state(dict):
            {"feature": value,...}
        - features(list): 如果为空则按照feature_bounds进行筛选并归一化
            {"feature0","feature1",...}
    Returns:
        - state(dict):
            除去features中没有的多余征，并且按照features调整顺序
    Options:
        - bounds(dict):如果不为空，输出按范围归一化后的状态
        - objtype(str):设置输出数据的对象类型
            - "dict"
            - "array"
    
    '''

    state_ = dict()
    ##按顺序过滤多余特征
    for f in features:
        state_[f] = state[f]

    ## 是否归一化
    if bounds:
        state_ = norm_state(state,bounds)

    #输出的数据类型
    if objtype == "dict":
        return state_
    if objtype == "array":
        return dict2array(state_)


def norm_state(state, bounds):
    """
    将输入的状态按bounds进行归一化
    Inputs:
        - state(dict):
            {"feature": value,...}
        - features_bounds(dict): 
            {"feature": [lower, upper],...}
    Returns:
        - state(dict):
            按照bounds诡异化后的state
    """
    state_ = dict()
    for f in bounds.keys():
        if f == 'heading-deg':
            if state[f] > 180.0:
                state_[f] = state[f] - 360.0
            else:
                state_[f] = state[f]
        bound = bounds[f]
        state_[f] = state[f]*2.0 / (bound[1]-bound[0])
    return state_

def dict2array(dict_):
    '''
    字典转化为numpy浮点数组
    '''
    return np.array(list(dict_.values())).astype(np.float)



def load_target_state(filedir):
    '''
        加载targetstate数据
        ---
        Inputs:
            - filedir: 存储轨迹的文件夹,最后不加 "/"
        Returns:
            - target_state:
                [
                    target_state_runway,
                    target_state_climbing, 
                    target_state_cruise
                ]
    '''
    target_state_runway = pd.read_csv( filedir + "/" + "trajectory_runway.csv")
    target_state_climbing = pd.read_csv( filedir + "/" + "trajectory_climbing.csv")
    target_state_cruise = pd.read_csv(filedir + "/" + "trajectory_cruise.csv")

    target_state = [target_state_runway,
                    target_state_climbing, target_state_cruise]
    return target_state

def get_target_state(state, target_states):
    '''
    从所有的target_state中抽取出最近的targetstate 
    ---
    Inputs:
        - state(dict):当前状态
        - target_states(dataframe): 所有的目标状态序列
    Returns：
        - target_state(dict): 选出的目标状态
    '''
    fly_mode = int(state["altitude"] > 23) + int(state["altitude"] > 6000)

    state_df = pd.DataFrame.from_dict([state])
    tmp = (target_states[fly_mode] - state_df)**2
    idx = tmp.sum(axis=1).idxmin()

    return target_states[fly_mode].loc[idx].to_dict()

def action2frame(action):
    '''
    转化action为控制帧
    Inputs:
        - action(obj):
            可以为numpy数组,tuple
    Returns:
        - action_frame(str):
            "%f,%f,%f,%f,%f\n"
    '''
    if type(action) == np.ndarray:
        a_ = tuple(action.tolist())
    if type(action) == tuple :
        a_ = action
    
    a_frame = "%f,%f,%f,%f,%f\n" % a_
    return a_frame

if __name__ == "__main__":
    
    bounds = {"1":[1,2],"2":[3,4]}
    print(bounds_dict2arr(bounds))
