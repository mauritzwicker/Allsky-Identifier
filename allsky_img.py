##########################################
'''
author: @mauritzwicker
date: 11.12.2020

Purpose: to create the allsky object for our images
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

import parameters
import functions
# import functions
# import parameters
########

########
class Allsky:
    def __init__(self, p1):
        #load the parameters
        self.params = parameters.Parameters_allsky()

        #define image name (self defined) and file locations
        self.image_name = p1
        self.file_location = self.params.image_full_PATH

        #we load the pickle file and correct for the darkcurrent
        self.img_obj_dict = functions.get_img(self.file_location, self.params.dark_location)
        #we get the date
        self.dt_string = (str(self.img_obj_dict['year']) + '/' + str(self.img_obj_dict['month']) + '/' +
                        str(self.img_obj_dict['day']) + ' ' + str(self.img_obj_dict['hour']) + ':' +
                        str(self.img_obj_dict['minute']) + ':' + str(self.img_obj_dict['second']))
        #create the observer and specify info
        self.observer = functions.create_observer(self.params, self.dt_string)

        #define image properties (image center)
        # self.x0 = self.img_obj_dict['xml_counts'].shape[1]/2
        # self.y0 = self.img_obj_dict['xml_counts'].shape[0]/2
        self.x0 = self.params.x0_define
        self.y0 = self.params.y0_define



        #check which stars are possibly visible in this image
        #call it awake_stars (ie above horizong)
        self.check_stars_visible()
        print('Found {0} awake stars in references!'.format(self.num_awake_stars))
        print('Found {0} asleep stars in references.'.format(self.num_sleep_stars))

        #now for each we know the altitude and azimuth and we want to convert this to a real image x,y position
        self.locate_awake_stars()


        pass


    def locate_awake_stars(self):
        # loop through all awake_stars and save their alt, az, x, y
        self.position_awake_stars_lit = {}    #the position of the awake stars (x-y, alt-az) from the literature alt-az values
        for star in self.awake_stars.keys():
            print(star)
            #get the ephem object
            star_obj = self.awake_stars[star]
            #get the alt,az,x,y from this object
            star_alt, star_az, star_x, star_y = functions.convert_coordinates_to_pix(star_obj, self.x0, self.y0, self.params)
            self.position_awake_stars_lit[str(star)] = [star_alt, star_az, star_x, star_y]
        print('Done Locating Awake Stars')

    def check_stars_visible(self):
        #to check which of the stars in params.stars_reference are visible (altitude >10deg)
        #dictionary to hold the ephem objects for our awake stars
        self.awake_stars = {}
        self.num_awake_stars = 0
        self.num_sleep_stars = 0
        for star in self.params.stars_reference:
            star_check = functions.check_star_awake(star, self.observer, self.params)
            if star_check == False:
                self.num_sleep_stars +=1
                continue
            else:
                self.num_awake_stars +=1
                self.awake_stars[str(star)] = star_check
        #the 'output' is a dict of the stars available with the psem object

########
