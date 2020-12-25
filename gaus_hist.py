##########################################
'''
author: @mauritzwicker
date: 14.12.2020

Purpose: to show us the format to use for files

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
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from numpy import log as ln

import photoutils_fct
########




########

def show_img_hist_normalized(img, show_question):
    ''' to show the image with the x and y histograms (for normalized counts data)'''
    rows = img.shape[0]
    cols = img.shape[1]

    #to get plot on yax
    yax_hist_y = []
    yax_hist_x = []
    for i in range(0, rows):
        row = img[i, :]
        # print(np.sum(row))
        yax_hist_y.append(np.sum(row))
        yax_hist_x.append(i)

    #to get plot on xax
    xax_hist_y = []
    xax_hist_x = []
    for i in range(0, cols):
        col = img[:, i]
        # print(np.sum(col))
        xax_hist_y.append(np.sum(col))
        xax_hist_x.append(i)

    yax_hist_y = (yax_hist_y-min(yax_hist_y))/(max(yax_hist_y)-min(yax_hist_y))
    xax_hist_y = (xax_hist_y-min(xax_hist_y))/(max(xax_hist_y)-min(xax_hist_y))

    sizeyfig = 6
    sizexfig = (sizeyfig/len(img[:,0])*len(img[0,:]))
    #figsize is x, y, so if we say we want y to be 6, then x is (6/len(img[:,0])*len(img[0,:])
    if show_question:
        fig = plt.figure(constrained_layout=False, figsize=(sizexfig, sizeyfig))
        spec = fig.add_gridspec(3, 3,wspace=0, hspace=0)
        ax1 = fig.add_subplot(spec[:-1,0])
        # ax1.set_title('ax1')
        ax1.barh(yax_hist_x[::-1], yax_hist_y)
        # ax1.set_yticks([])
        ax2 = fig.add_subplot(spec[:-1:, 1:])
        # ax2.set_title('ax2')
        ax2.imshow(img, aspect = 'auto')
        ax2.axis('off')
        ax3 = fig.add_subplot(spec[-1, 1:])
        # ax3.set_title('ax3')
        ax3.bar(xax_hist_x, xax_hist_y)
        # ax3.set_xticks([])
        ax3.yaxis.tick_right()
        ax2.set_aspect('equal')
        # plt.tight_layout()
        # plt.suptitle('NORMALIZED IMAGE')
        plt.show()

    # now we want to return the x and y hist arrays so that we can fit them
    return([yax_hist_x, yax_hist_y, xax_hist_x, xax_hist_y])

def show_img_hist_full(img):
    ''' to show the image with the x and y histograms (for normalized counts data)'''
    rows = img.shape[0]
    cols = img.shape[1]

    #to get plot on yax
    yax_hist_y = []
    yax_hist_x = []
    for i in range(0, rows):
        row = img[i, :]
        # print(np.sum(row))
        yax_hist_y.append(np.sum(row))
        yax_hist_x.append(i)

    #to get plot on xax
    xax_hist_y = []
    xax_hist_x = []
    for i in range(0, cols):
        col = img[:, i]
        # print(np.sum(col))
        xax_hist_y.append(np.sum(col))
        xax_hist_x.append(i)

    sizeyfig = 6
    sizexfig = (sizeyfig/len(img[:,0])*len(img[0,:]))
    #figsize is x, y, so if we say we want y to be 6, then x is (6/len(img[:,0])*len(img[0,:])
    fig = plt.figure(constrained_layout=False, figsize=(sizexfig, sizeyfig))
    spec = fig.add_gridspec(3, 3,wspace=0, hspace=0)
    ax1 = fig.add_subplot(spec[:-1,0])
    # ax1.set_title('ax1')
    ax1.barh(yax_hist_x, yax_hist_y)
    ax1.set_yticks([])
    ax2 = fig.add_subplot(spec[:-1:, 1:])
    # ax2.set_title('ax2')
    im2 = ax2.imshow(img, aspect = 'auto')
    ax2.axis('off')
    ax3 = fig.add_subplot(spec[-1, 1:])
    # ax3.set_title('ax3')
    ax3.bar(xax_hist_x, xax_hist_y)
    ax3.set_xticks([])
    ax3.yaxis.tick_right()
    ax2.set_aspect('equal')
    # plt.tight_layout()
    # plt.suptitle('NORMALIZED IMAGE')
    fig.colorbar(im2, orientation='vertical')
    plt.show()

def gaus(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def fit_gaus(x_data, y_data, plottrue=False):
    x = x_data
    y = y_data
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gaus, x, y, p0=[min(y), max(y), mean, sigma])

    if plottrue:
        plt.plot(x,y,'b+:',label='data')
        plt.plot(x,gaus(x,*popt),'ro:',label='fit')
        plt.legend()
        # plt.title('Gauß ax1, (ROWS OF IMSHOW SUMMED)')
        plt.xlabel('wavelength Angstrom')
        plt.ylabel('')
        plt.show()

    #    so popt is our returned fit parameters: [H, A, x0, sigma]
    #    so our function is: y = H + A * np.exp( - (x-x0)**2 / (2*sigma**2))
    #    also ist unsere standard abweichung: sigma
    sigma = abs(popt[3])
    # print(sigma)
    fwhm_mes = 2 * math.sqrt(2 * ln(2)) * sigma
    # print(fwhm_mes)
    #so we return (FWHM, and popt)
    return(fwhm_mes, popt)

def fit_fins(rowscols_data, show_question):
    ''' fit the peak data to find the peak and FWHM of gaussian '''
    #rowscols_data[0] is the y axis of imshow (ax1 in plots above) so the row #
    #rowscols_data[1] is the x axis of ax1 in plots above. so the summed value of that row
    #rowscols_data[2] is the x axis of imshow (ax3 in plots above) so the column #
    #rowscols_data[3] is the y axis of ax3 in plots above. so the summed value of that col
    #plot scatter of rows (ax1 above)

    # plt.scatter(rowscols_data[0], rowscols_data[1])
    # plt.show()

    # plt.scatter(rowscols_data[2], rowscols_data[3])
    # plt.show()

    #determine gaus fit for ax1 above, rows in imshow (2 dimension)
    fwhm_rows, fit_rows = fit_gaus(rowscols_data[0], rowscols_data[1], show_question)
    fit_rows = list(fit_rows)
    fit_rows.append(fwhm_rows)

    #determine gaus fit for ax3 above, cols in imshow (1 dimension)
    fwhm_cols, fit_cols = fit_gaus(rowscols_data[2], rowscols_data[3], show_question)
    fit_cols = list(fit_cols)
    fit_cols.append(fwhm_cols)

    #we return the data for rows, and cols where:
    #fit_rows = [H, A, x0, sigma, fwhm]
    #fit_cols = [H, A, x0, sigma, fwhm]
    return(fit_rows, fit_cols)






def test_perfectdata(star_d):
    # to test with perfect data:
    # input is the image as xml file (like from main.py)
    print('STARTING TEST')
    # star_d = image.img_obj_dict['xml_counts']       #star_data

    #mars
    x_min = 1610
    x_max = 1669
    y_min = 1580
    y_max = 1620
    #the position of the star in this frame
    obj_mars = star_d[y_min:y_max, x_min:x_max]
    bars_data = show_img_hist_normalized(obj_mars)
    rows_params, cols_params = fit_fins(bars_data)
    plt.imshow(obj_mars)
    plt.scatter(cols_params[2], rows_params[2])
    plt.scatter(cols_params[2] + (cols_params[4]/2), rows_params[2])
    plt.scatter(cols_params[2] - (cols_params[4]/2), rows_params[2])
    plt.scatter(cols_params[2], rows_params[2] + (rows_params[4]/2))
    plt.scatter(cols_params[2], rows_params[2] - (rows_params[4]/2))
    plt.title('Mars')
    plt.show()
    print('Mars')
    print('FWHM {0} and {1}'.format(rows_params[4], cols_params[4]))
    print()
    print()

    #now with background correction
    x_min = 1610
    x_max = 1669
    y_min = 1580
    y_max = 1620
    #the position of the star in this frame
    obj_mars = star_d[y_min:y_max, x_min:x_max]
    obj_mars = photoutils_fct.background_noise_est(obj_mars)
    bars_data = show_img_hist_normalized(obj_mars)
    rows_params, cols_params = fit_fins(bars_data)
    plt.imshow(obj_mars)
    plt.scatter(cols_params[2], rows_params[2])
    plt.scatter(cols_params[2] + (cols_params[4]/2), rows_params[2])
    plt.scatter(cols_params[2] - (cols_params[4]/2), rows_params[2])
    plt.scatter(cols_params[2], rows_params[2] + (rows_params[4]/2))
    plt.scatter(cols_params[2], rows_params[2] - (rows_params[4]/2))
    plt.title('Mars background corrected')
    plt.show()
    print('Mars background corrected')
    print('FWHM {0} and {1}'.format(rows_params[4], cols_params[4]))
    print()
    print()



    #************************
    #almach
    x_min = 1575
    x_max = 1585
    y_min = 1154
    y_max = 1164
    #the position of the star in this frame
    obj_almach = star_d[y_min:y_max, x_min:x_max]
    obj_almach = photoutils_fct.background_noise_est(obj_almach)
    bars_data = show_img_hist_normalized(obj_almach)
    rows_params, cols_params = fit_fins(bars_data)
    plt.imshow(obj_almach)
    plt.scatter(cols_params[2], rows_params[2])
    plt.scatter(cols_params[2] + (cols_params[4]/2), rows_params[2])
    plt.scatter(cols_params[2] - (cols_params[4]/2), rows_params[2])
    plt.scatter(cols_params[2], rows_params[2] + (rows_params[4]/2))
    plt.scatter(cols_params[2], rows_params[2] - (rows_params[4]/2))
    plt.title('Almach')
    plt.show()
    print('Almach')
    print('FWHM {0} and {1}'.format(rows_params[4], cols_params[4]))
    print()
    print()
    #now with background correction
    x_min = 1575
    x_max = 1585
    y_min = 1154
    y_max = 1164
    #the position of the star in this frame
    obj_almach = star_d[y_min:y_max, x_min:x_max]
    bars_data = show_img_hist_normalized(obj_almach)
    rows_params, cols_params = fit_fins(bars_data)
    plt.imshow(obj_almach)
    plt.scatter(cols_params[2], rows_params[2])
    plt.scatter(cols_params[2] + (cols_params[4]/2), rows_params[2])
    plt.scatter(cols_params[2] - (cols_params[4]/2), rows_params[2])
    plt.scatter(cols_params[2], rows_params[2] + (rows_params[4]/2))
    plt.scatter(cols_params[2], rows_params[2] - (rows_params[4]/2))
    plt.title('Almach background corrected')
    plt.show()
    print('Almach background corrected')
    print('FWHM {0} and {1}'.format(rows_params[4], cols_params[4]))
    print()
    print()



    #************************
    #menkalinan
    x_min = 1238
    x_max = 1252
    y_min = 825
    y_max = 836
    #the position of the star in this frame
    obj_menka = star_d[y_min:y_max, x_min:x_max]
    obj_menka = photoutils_fct.background_noise_est(obj_menka)
    bars_data = show_img_hist_normalized(obj_menka)
    rows_params, cols_params = fit_fins(bars_data)
    plt.imshow(obj_menka)
    plt.scatter(cols_params[2], rows_params[2])
    plt.scatter(cols_params[2] + (cols_params[4]/2), rows_params[2])
    plt.scatter(cols_params[2] - (cols_params[4]/2), rows_params[2])
    plt.scatter(cols_params[2], rows_params[2] + (rows_params[4]/2))
    plt.scatter(cols_params[2], rows_params[2] - (rows_params[4]/2))
    plt.title('Menkalinan')
    plt.show()
    print('Menkalinan')
    print('FWHM {0} and {1}'.format(rows_params[4], cols_params[4]))
    print()
    print()
    #now with background correction
    x_min = 1238
    x_max = 1252
    y_min = 825
    y_max = 836
    #the position of the star in this frame
    obj_menka = star_d[y_min:y_max, x_min:x_max]
    bars_data = show_img_hist_normalized(obj_menka)
    rows_params, cols_params = fit_fins(bars_data)
    plt.imshow(obj_menka)
    plt.scatter(cols_params[2], rows_params[2])
    plt.scatter(cols_params[2] + (cols_params[4]/2), rows_params[2])
    plt.scatter(cols_params[2] - (cols_params[4]/2), rows_params[2])
    plt.scatter(cols_params[2], rows_params[2] + (rows_params[4]/2))
    plt.scatter(cols_params[2], rows_params[2] - (rows_params[4]/2))
    plt.title('Menkalinan background corrected')
    plt.show()
    print('Menkalinan background corrected')
    print('FWHM {0} and {1}'.format(rows_params[4], cols_params[4]))
    print()
    print()



    print('TEST DONE')

########





if __name__ == '__main__':
    t_start = time.perf_counter()
    pid = os.getpid()
    process = psutil.Process(pid)


    ########
    # CODE
    ########


    #Done
    t_fin = time.perf_counter()
    print('\n \n \n********* Process-{0} over ************'.format(pid))
    print('Runtime: {0} seconds'.format(t_fin - t_start))
    p_mem = process.memory_info().rss
    print('Memory used: {0} bytes ({1} GB) '.format(p_mem, p_mem/(1e9)))

