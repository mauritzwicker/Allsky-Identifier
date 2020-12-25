##########################################
'''
author: @mauritzwicker
date: 11.12.2020
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

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import parameters
import allsky_img
import photoutils_fct
import gaus_hist

if __name__ == '__main__':
    t_start = time.perf_counter()
    pid = os.getpid()
    process = psutil.Process(pid)


    ########
    params = parameters.Parameters_allsky()
    #first we want to create the image object:
    image_NAME = params.name_IMG
    image = allsky_img.Allsky(image_NAME)

    if params.plot_awake_stars:
        NUM_COLORS = len(image.position_awake_stars_lit.keys())
        cm = plt.get_cmap('gist_rainbow')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

        ax.imshow(image.img_obj_dict['xml_counts'], cmap = 'gray')
        ax.scatter(params.x0_define, params.y0_define, marker = 'x', label = 'zenith')
        # plt.show()
        for star in image.position_awake_stars_lit.keys():
            pos_star = image.position_awake_stars_lit[star]
            star_x = pos_star[2]
            star_y = pos_star[3]
            ax.scatter(star_x, star_y, label = star, alpha = 0.5)
        # plt.legend()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Stars',  ncol=2)
        plt.title('-- {0} Allsky Image -- \n {1} \n {2}{3} - {4}{5} \n alt: {6}m '.format(image.image_name,
            image.dt_string, params.observer_lat, params.observer_lat_letter,
            params.observer_long, params.observer_long_letter,
            round(params.observer_elevation, 2)))
        plt.show()

    print('\n\n\n\n')

    #cut the array for a star:
    for star in image.position_awake_stars_lit.keys():
        print(star)
        pos_star = image.position_awake_stars_lit[star]


        #shoudl create star object
        x_min = int(pos_star[2])-params.deviation_position
        x_max = int(pos_star[2])+params.deviation_position
        y_min = int(pos_star[3])-params.deviation_position
        y_max = int(pos_star[3])+params.deviation_position
        #the position of the star in this frame
        new_rel_pos = [int(pos_star[2]-x_min), int(pos_star[3] - y_min)]

        #cut the image into the area we defined
        star_d = image.img_obj_dict['xml_counts']       #star_data
        # star_d_c = star_d[x_min:x_max, y_min:y_max]     #star_data_cut_
        star_d_c = star_d[y_min:y_max, x_min:x_max]

        #maybe take away background?
        # star_d_c = photoutils_test.background_noise_est(star_d_c)

        #for test with perfectly close data:
        # gaus_hist.test_perfectdata(star_d)
        # quit()

        # gaus_hist_test.show_img_hist_full(star_d)
        # gaus_hist_test.show_img_hist_normalized(star_d)
        # gaus_hist_test.show_img_hist_full(obj_mars)
        show_hist = False
        bars_data = gaus_hist.show_img_hist_normalized(star_d_c, show_hist)

        # #first second dim of imshow
        # # plt.barh(bars_data[0], bars_data[1])
        # plt.scatter(bars_data[0], bars_data[1])
        # plt.show()

        # #first first dim of imshow
        # # plt.bar(bars_data[2], bars_data[3])
        # plt.scatter(bars_data[2], bars_data[3])
        # plt.show()

        #now we want to fit a gaußian:
        show_hist = False
        rows_params, cols_params = gaus_hist.fit_fins(bars_data, show_hist)
        #rows_params = [H, A, x0, sigma, fwhm] for rows fit
        #cols_params = [H, A, x0, sigma, fwhm] for cols fit


        '''
        plt.imshow(star_d_c)
        plt.scatter(cols_params[2], rows_params[2])
        plt.scatter(cols_params[2] + (cols_params[4]/2), rows_params[2])
        plt.scatter(cols_params[2] - (cols_params[4]/2), rows_params[2])
        plt.scatter(cols_params[2], rows_params[2] + (rows_params[4]/2))
        plt.scatter(cols_params[2], rows_params[2] - (rows_params[4]/2))
        plt.show()

        plt.imshow(star_d_c)
        plt.scatter(cols_params[2], rows_params[2])
        plt.scatter(cols_params[2] + (cols_params[4]/2), rows_params[2] + (rows_params[4]/2))
        plt.scatter(cols_params[2] + (cols_params[4]/2), rows_params[2] - (rows_params[4]/2))
        plt.scatter(cols_params[2] - (cols_params[4]/2), rows_params[2] + (rows_params[4]/2))
        plt.scatter(cols_params[2] - (cols_params[4]/2), rows_params[2] - (rows_params[4]/2))
        plt.show()

        a = rows_params[2] - (rows_params[4]/2)
        b = rows_params[2] + (rows_params[4]/2)
        c = cols_params[2] - (cols_params[4]/2)
        d = cols_params[2] + (cols_params[4]/2)
        print(a, b, c, d)
        a = int(a)
        b = int(b)
        c = int(c)
        d = int(d)
        ar_fwhm = star_d_c[a:b, c:d]
        print(np.sum(ar_fwhm))
        plt.imshow(ar_fwhm)
        plt.show()
        a = int(a)
        b = int(b)+1
        c = int(c)
        d = int(d)+1
        print(a, b, c, d)
        # ar_fwhm = star_d_c[rows_params[2] - (rows_params[4]/2):rows_params[2] + (rows_params[4]/2), cols_params[2] - (cols_params[4]/2): cols_params[2] + (cols_params[4]/2)]
        ar_fwhm = star_d_c[a:b, c:d]
        print(np.sum(ar_fwhm))
        plt.imshow(ar_fwhm)
        plt.show()
        '''





        print('FWHM {0} and {1}'.format(rows_params[4], cols_params[4]))
        print()
        print()

        # quit()





        #




    ########


    #Done
    t_fin = time.perf_counter()
    print('\n \n \n********* Process-{0} over ************'.format(pid))
    print('Runtime: {0} seconds'.format(t_fin - t_start))
    p_mem = process.memory_info().rss
    print('Memory used: {0} bytes ({1} GB) '.format(p_mem, p_mem/(1e9)))
