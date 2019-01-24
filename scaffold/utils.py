# -*- coding: utf-8 -*-
import datetime
from geographiclib.geodesic import Geodesic


def get_azimuth(lat1, lon1, lat2, lon2):
    '''
    给定两个点的经纬度，计算第二个点相对于第一个点的方位角
    ---
    Inputs:
        lat1,lon1(latitude,longitude) #第一个点的经纬度坐标
        lat2,lon2(latitude,longitude) #第二个点的经纬度坐标
    Returns：
        angle #方位角
    '''
    outmask = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
    angle = outmask['azi1']
    return angle


def gettime():
    """
    return a time str for now
    """
    now = datetime.datetime.now()
    date = "%s-%s-%s_%s-%s-%s" % (
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    return date
