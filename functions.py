##########################################
'''
author: @mauritzwicker
date: 11.12.2020

Purpose: to be used as a collection of useful functions

'''
##########################################

########
#IMPORTS
import time
import psutil
import os

import cv2
import numpy as np
from xml.etree import ElementTree
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import math
import ephem
import datetime
from datetime import datetime
from matplotlib.patches import Circle
from skyfield.api import load, utc
import pandas as pd

# from conversion_coord import *

########

def get_img(path, dark_loc):
    #to load the pickle file for the object (or whatever file format!)
    with open(path,'rb') as file:
        object_file = pickle.load(file)
    with open(dark_loc, 'rb') as dfile:
        dark_obj = np.load(dark_loc)
    #correct for dc
    xmlcounts_dc = np.subtract(object_file['xml_counts'], dark_obj)
    object_file['xml_counts'] = xmlcounts_dc

    #dark current is so small it can be neglected actually
    return(object_file)

def create_observer(params, obs_datetime):
    #to create the ephem observer object
    observer = ephem.Observer()
    observer.lat = params.observer_lat
    observer.lon = params.observer_long
    observer.elevation = params.observer_elevation
    observer.date = obs_datetime
    return(observer)

def check_star_awake(star_name, observer, params):
    #check whether the star is awake (ie. abover a specific threshold')
    star_obj = ephem.star(star_name)
    star_obj.compute(observer)
    if star_obj.neverup:
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(star_name)
        # print('Star NOT visible at your location')
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print()
    #check if the altitude is over 10 degrees (in radians)
    if (star_obj.alt + 0.0) >= (params.awake_thresh * math.pi/180):
        # print('{0} is AWAKE'.format(star_name))
        return(star_obj)
    else:
        # print('{0} is sleeping'.format(star_name))
        return(False)

def convert_coordinates_to_pix(obj, x_mid, y_mid, params):
    #to convert the coordinates from alt, aximuth to pixel values
    obj_altitude = obj.alt + 0.0
    obj_azimuth = obj.az + 0.0


    #convert this into x, y with out converion
    obj_coords = xy_from_azalt(obj_azimuth, obj_altitude, x_mid, y_mid, params.factors_ang_pix)

    obj_x = obj_coords['x']
    obj_y = obj_coords['y']

    return(obj_altitude, obj_azimuth, obj_x, obj_y)










#Conversions


def xyrel_given_xy(x1, y1):
    x_rel = x1- x0
    y_rel = y1 - y0
    return(x_rel, y_rel)

def rpix_given_xyrel(xrel, yrel):
    r_pixs = np.sqrt(xrel**2 + yrel**2)
    return(r_pixs)

def rpixLIT_given_rpixXY(rpixs):
    m = 1.0227338615139565
    b = -11.181533548204177
    rpix_lit = m * rpixs + b
    return(rpix_lit)

def alt_given_rpixLIT(rpix):
    rarcmin = rpix * factors_ang_pix
    rrad = rarcmin / 3437.75
    alt = math.pi/2 - rrad
    return(alt)

def rpixLIT_given_alt(alt):
    rrad = math.pi/2 - alt
    rarcmin = rrad * 3437
    rpix = rarcmin / factors_ang_pix
    return(rpix)

def rpixXY_give_rpixLIT(rpix):
    m = 1.0227338615139565
    b = -11.181533548204177
    rpix_xy = (rpix - b)/m
    return(rpix_xy)




######################### INSIDE FUNCTIONS FOR THETA #########################

def sign_correct_theta_given_xyrel(theta, xrel, yrel):
    if xrel <= 0:
        return(theta + math.pi)
    elif xrel >=0:
        if yrel <= 0:
            return(theta + 2*math.pi)
        else:
            return(theta)

def theta_given_xyrel(xrel, yrel):
    theta = math.atan(yrel/xrel)
    theta_corr = sign_correct_theta_given_xyrel(theta, xrel, yrel)
    return(theta_corr)

def az_given_theta(theta):
    m = -0.9927116594468144
    b = 5.306998457927701
    az = m * theta + b
    if az >= 2*math.pi:
        az -= 2*math.pi
    elif az <= 0:
        az += 2*math.pi
    return(az)

def theta_given_az(az):
    #To get the theta, angle from y=0 line, given the azimuth from ephem
    m = -0.9927116594468144
    b = 5.306998457927701
    theta = (az-b)/m
    if theta <= 0:
        theta += 2* math.pi
    return(theta)

def thet_transfo_2(x, y, theta):
    if theta >=0 and theta <= math.pi/2:
        return(abs(x), abs(y))
    elif theta >=math.pi/2 and theta <= math.pi:
        return(-1*abs(x), abs(y))
    elif theta >=math.pi and theta <= 3 * math.pi/2:
        return(-1*abs(x), -1*abs(y))
    elif theta >=3 * math.pi/2 and theta <= 2 * math.pi:
        return(abs(x), -1*abs(y))

def xy_given_xyrel(xrel, yrel):
    x = xrel + x0
    y = yrel + y0
    return(x,y)

def rpix_given_alt(alt):
    #To get the distance from center in pixels, given the altitude from ephem
    rpix_lit = rpixLIT_given_alt(alt)
    rpix_xy = rpixXY_give_rpixLIT(rpix_lit)
    #when we have the angle we can then get the xy values
    return(rpix_xy)




################### USER FUNCTIONS #######################
#*** TO GET ALT/AZ with XY
def alt_given_xy(x, y):
    #To get the altitude we expect in ephem given the xy coordinates on the image
    x_rel, y_rel = xyrel_given_xy(x,y)
    r_pixs = rpix_given_xyrel(x_rel, y_rel)
    rpix_lit = rpixLIT_given_rpixXY(r_pixs)
    alt = alt_given_rpixLIT(rpix_lit)
    return(alt)

def az_given_xy(x, y):
    #To get the azimuth we expect in ephem, given the xy coordinates on the image
    x_rel, y_rel = xyrel_given_xy(x,y)
    theta_xy = theta_given_xyrel(x_rel, y_rel)
    az = az_given_theta(theta_xy)
    return(az)


#*** TO GET XY with ALT/AZ
def xy_given_altaz(alt, az):
    #to get the x and y values given alt and az
    rpix = rpix_given_alt(alt)
    theta = theta_given_az(az)

    x_rel = rpix * math.cos(theta)
    y_rel = rpix * math.sin(theta)
    x_rel, y_rel = thet_transfo_2(x_rel, y_rel, theta)

    x,y = xy_given_xyrel(x_rel, y_rel)

    return(x,y)


# xy_given_altaz -> returns (x, y)
# az_given_xy -> returns (azimuth in radians)
# alt_given_xy -> returns (altitude in radians)


# GET AZ, ALT from x,y
def azalt_from_xy(x, y, x_mid, y_mid, fac_pix):
    global x0
    x0 = x_mid
    global y0
    y0 = y_mid
    global factors_ang_pix
    factors_ang_pix = fac_pix

    az = az_given_xy(x, y)
    alt = alt_given_xy(x, y)

    dict_star = {}
    dict_star['x'] = x
    dict_star['y'] = y
    dict_star['alt'] = alt
    dict_star['az'] = az
    return(dict_star)

# GET x,y, from AZ, ALT
def xy_from_azalt(az, alt, x_mid, y_mid, fac_pix):
    global x0
    x0 = x_mid
    global y0
    y0 = y_mid
    global factors_ang_pix
    factors_ang_pix = fac_pix

    x, y = xy_given_altaz(alt, az)
    dict_star = {}
    dict_star['alt'] = alt
    dict_star['az'] = az
    dict_star['x'] = x
    dict_star['y'] = y
    return(dict_star)



########
# CODE
########
