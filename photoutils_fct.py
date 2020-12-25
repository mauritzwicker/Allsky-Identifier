##########################################
'''
author: @mauritzwicker
date: 11.12.2020

Purpose: to show us the format to use for files

'''
##########################################

########
#IMPORTS
import time
import psutil
import os


import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import pandas as pd
import seaborn as sns
import datetime
from datetime import datetime
import math
import ephem
import warnings

warnings.filterwarnings("ignore")

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import DAOStarFinder
from astropy.stats import mad_std
from photutils import aperture_photometry, CircularAperture
from astropy.stats import biweight_location
from astropy.stats import mad_std
from astropy.stats import sigma_clipped_stats
from photutils import make_source_mask
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground
from astropy.stats import sigma_clipped_stats
from photutils import find_peaks
from photutils import centroid_com, centroid_1dg, centroid_2dg, centroid_sources
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from astropy.visualization import simple_norm



########




########
#TASK 1 BACKGROUND
def background_noise_est(img_data):
    '''
        We want to get information about the Background and Noise of the image
        https://photutils.readthedocs.io/en/stable/background.html
    '''
    #look at the image we are doing the estimation for
    norm = ImageNormalize(stretch=SqrtStretch())
    # plt.imshow(img_data, norm=norm, origin='lower', cmap='Greys_r', interpolation='nearest')
    # plt.title('Image Original')
    # plt.show()

    #print the image median and biweight location as a reference to background level (they are larger though)
    # print(np.median(img_data))
    # print(biweight_location(img_data))
    #print the image std of background estimate using (is is larger though)
    # print(mad_std(img_data))

    #So to get the real value
    #we use sigma_clippins (read documentation) but first want to mask our sources, for an accurate value
    #we set nsigma, npixels, and dilate_size:
    #mask source docu: https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.make_source_mask.html#photutils.segmentation.make_source_mask
    mask = make_source_mask(img_data, nsigma=3, npixels=1)
    #so npixels is how many pixels for something to be considered a source
    #sigma is amount of sigma a pixel needs to be above background to be considered for source
    mean, median, std = sigma_clipped_stats(img_data, sigma=3.0, mask=mask)
    # print((mean, median, std))

    #to get the actual background:
    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg = Background2D(img_data, (5, 5), filter_size=(3, 3),

    sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    #this give us our background of
    print('BACKGROUND IS...')
    print(bkg.background_median)
    print(bkg.background_rms_median)
    # which looks like this:
    # plt.imshow(bkg.background, origin='lower', cmap='Greys_r',interpolation='nearest')
    # plt.title('Image Background')
    # plt.show()
    # plt.imshow(img_data - bkg.background, norm=norm, origin='lower', cmap='Greys_r', interpolation='nearest')
    # plt.title('Image corrected for Background')
    # plt.show()

    return(img_data - bkg.background)
    # return(img_data - bkg.background, bkg.background_median)

# Part 2: Source Detection
def source_detection(img_data):
    '''
        This is to detect stars (or object of differen thresholds)
        And to determine local peaks
        https://photutils.readthedocs.io/en/stable/detection.html
    '''
    #We will estimate the background and background noise using sigma-clipped statistics:
    mean, median, std = sigma_clipped_stats(img_data, sigma=3.0)
    # print((mean, median, std))
    #ALREADY BACKGROUND CORRECTED

    #Now we will subtract the background and use an instance of DAOStarFinder to find the
    #stars in the image that have FWHMs of around 3 pixels and have peaks approximately
    #5-sigma above the background. Running this class on the data yields an astropy Table containing
    #the results of the star finder:

    # so increaseding fwhm doesn;t change anythin until at some point it observed more values (s0 don't have too high)
    # high sigma means no stars detected
    # low sigma means everything detected
    # so our plan is to use ephem to find how bright the star is supposed to be -> select std and fwhm
    # for i in range(1, 20):
    #     print('fwhm = ', 3)
    #     print('sigma = ', i)
    #     daofind = DAOStarFinder(fwhm=3, threshold=i*std)
    #     sources = daofind(img_data)
    #     print(sources)

    # daofind = DAOStarFinder(fwhm=3, threshold=5*std)
    # sources = daofind(img_data)
    # plt.imshow(img_data)
    # plt.show()

    thresh = 50
    while thresh >=3:
        fwhm = 8
        while fwhm >=3:
            daofind = DAOStarFinder(fwhm=fwhm, threshold=thresh*std)
            sources = daofind(img_data)

            if sources == None:
                fwhm -= 1
            else:
                print(fwhm, thresh)
                return(sources)
        thresh -= 1
    print('wtf')

    return(None)


    # daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
    # sources = daofind(img_data)
    # if sources == None:
    #     return(None)
    # for col in sources.colnames:
    #     sources[col].info.format = '%.8g'  # for consistent table output
    # # print(sources)

    # #now we plot the star finder
    # positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    # apertures = CircularAperture(positions, r=4.)
    # norm = ImageNormalize(stretch=SqrtStretch())
    # plt.imshow(img_data, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
    # apertures.plot(color='blue', lw=1.5, alpha=0.5)
    # plt.title('Star Finder Results')
    # plt.show()

    # #we can use find peaks to find local peaks in an image that are above a specified threshold value.
    # #Peaks are the local maxima above a specified threshold that are separated by a specified minimum number of pixels.
    # # look at peaks documentation for more info
    # #https://photutils.readthedocs.io/en/stable/api/photutils.detection.find_peaks.html#photutils.detection.find_peaks
    # mean, median, std = sigma_clipped_stats(img_data, sigma=3.0)
    # threshold = median + (5. * std)
    # tbl = find_peaks(img_data, threshold, box_size=11)
    # tbl['peak_value'].info.format = '%.8g'  # for consistent table output
    # print(tbl[:10])  # print only the first 10 peaks

    # #to plot the peaks positions
    # positions = np.transpose((tbl['x_peak'], tbl['y_peak']))
    # apertures = CircularAperture(positions, r=5.)
    # norm = simple_norm(img_data, 'sqrt', percent=99.9)
    # plt.imshow(img_data, cmap='Greys_r', origin='lower', norm=norm, interpolation='nearest')
    # apertures.plot(color='#0547f9', lw=1.5)
    # plt.xlim(0, img_data.shape[1]-1)
    # plt.ylim(0, img_data.shape[0]-1)
    # plt.title('Peak Finder Results')
    # plt.show()

    # # print(sources.colnames)
    # # # print(sources.columns())
    # # quit()

    # xxx = list(sources['xcentroid'])
    # yyy = list(sources['ycentroid'])
    # # plt.imshow(img_data, cmap = 'gray')
    # # plt.scatter
    # return(sources)



# Part 3: Centroids
def centroid(img_data):
    '''
        To find the centroids of a source (or multiple images)
        https://photutils.readthedocs.io/en/stable/centroids.html
    '''
    #we are not subtracting the background (but one should!)
    x1, y1 = centroid_com(img_data)
    x2, y2 = centroid_1dg(img_data)
    x3, y3 = centroid_2dg(img_data)
    print((x1, y1))
    print((x2, y2))
    print((x3, y3))

    fig, ax = plt.subplots(1, 1)
    ax.imshow(img_data, origin='lower', interpolation='nearest')
    marker = '+'
    ms, mew = 15, 2.
    plt.plot(x1, y1, color='black', marker=marker, ms=ms, mew=mew)
    plt.plot(x2, y2, color='white', marker=marker, ms=ms, mew=mew)
    plt.plot(x3, y3, color='red', marker=marker, ms=ms, mew=mew)

    ax2 = zoomed_inset_axes(ax, zoom=6, loc=9)
    ax2.imshow(img_data, vmin=190, vmax=220, origin='lower',
               interpolation='nearest')
    ms, mew = 30, 2.
    ax2.plot(x1, y1, color='white', marker=marker, ms=ms, mew=mew)
    ax2.plot(x2, y2, color='black', marker=marker, ms=ms, mew=mew)
    ax2.plot(x3, y3, color='red', marker=marker, ms=ms, mew=mew)
    ax2.set_xlim(13, 15)
    ax2.set_ylim(16, 18)
    mark_inset(ax, ax2, loc1=3, loc2=4, fc='none', ec='0.5')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax.set_xlim(0, img_data.shape[1]-1)
    ax.set_ylim(0, img_data.shape[0]-1)
    plt.show('Single Centroid')




########

def do_eval(name, vals, object_file):
    star_name = name
    full_data = object_file['xml_counts']
    star_x = vals[2]
    star_y = vals[3]
    variation_val = 30      #maybe try with like iterative process to go out more and more
    cut_data = full_data[int(star_x) - variation_val:int(star_x) + variation_val, int(star_y) - variation_val:int(star_y) + variation_val]

    print(star_name)

    cut_data_woback = background_noise_est(cut_data)

    sources_detected = source_detection(cut_data_woback)

    #this is our astropy table
    #columns;
    '''id, xcentroid, ycentroid, sharpness, roundness1, roundness2, npix, sky, peak, flux, mag'''
    if sources_detected == None:
        print(star_name)
        print('none found')
        print()
    else:
        print(star_name)
        print(list(sources_detected['flux']))
        print(list(sources_detected['mag']))
        print(list(sources_detected['roundness1']))
        print(list(sources_detected['roundness2']))
        print(list(sources_detected['sharpness']))
        print(list(sources_detected['npix']))
        print()



if __name__ == '__main__':
    t_start = time.perf_counter()
    pid = os.getpid()
    process = psutil.Process(pid)


    ########
    #for the tests we just take an image and try:
    # Sirrah [star_alt, star_az, star_x, star_y]]
    # [0.8095796704292297, 4.574984550476074, 1818.761111278641, 1335.8668986857397]
    path = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/outputs/data_out/20201118230658_eval.pickle'
    with open(path,'rb') as file:
        object_file = pickle.load(file)

    star_objs = [['Sirrah', 0.8095796704292297, 4.574984550476074, 1818.761111278641, 1335.8668986857397],
    ['Caph', 1.0361664295196533, 5.383929252624512, 1802.6196535108706, 983.3952009410636],
    ['Schedar', 1.105888843536377, 5.279574394226074, 1760.1458910167203, 1018.2937623601481],
    ['Mirach', 1.0550212860107422, 4.472360134124756, 1681.3528677740774, 1257.3293939434827],
    ['Almach', 1.272154450416565, 4.443765640258789, 1586.9585875128535, 1160.3561088441334],
    ['Hamal', 1.030287742614746, 3.864910125732422, 1500.9290105930274, 1354.88946072825],
    ['Polaris', 0.873419463634491, 6.280432224273682, 1707.6236562941808, 640.3397186384411],
    ['Algol', 1.4045782089233398, 3.6857197284698486, 1452.8692086368537, 1124.1535760755921],
    ['Electra', 1.1294331550598145, 3.0436980724334717, 1274.0215169115072, 1226.7578356634285],
    ['Taygeta', 1.1354992389678955, 3.0395185947418213, 1275.5808600765845, 1223.1182176621214],
    ['Maia', 1.1335757970809937, 3.03403902053833, 1273.619041216694, 1222.99785317537],
    ['Merope', 1.1261225938796997, 3.030756711959839, 1269.849813122005, 1225.8609915980298],
    ['Alcyone', 1.1284600496292114, 3.019500255584717, 1268.388284353509, 1222.6104392420302],
    ['Atlas', 1.1269336938858032, 3.0041933059692383, 1264.4861453261146, 1220.326492241346],
    ['Aldebaran', 0.9578869938850403, 2.6841440200805664, 1115.5681317383996, 1197.9322154809638],
    ['Capella', 1.265444278717041, 1.5936181545257568, 1294.0297981440804, 896.6848161100869],
    ['Elnath', 1.0516595840454102, 2.1619374752044678, 1126.1086547706318, 1001.1301286749109],
    ['Menkalinan', 1.1369588375091553, 1.5062286853790283, 1242.8171458031134, 831.8140941947545],
    ['Alfirk', 0.7897443175315857, 5.793728351593018, 1898.4552995280712, 775.9622433788364],
    ['Alpheratz', 0.8095781207084656, 4.574986457824707, 1818.7624512911725, 1335.8668578273068],
    ['Mirfak', 1.5415561199188232, 5.059398651123047, 1488.228749673194, 1017.1904748756351]]
    # starsnames = ['Sirrah','Caph','Schedar','Mirach','Almach','Hamal','Polaris','Algol','Electra','Taygeta','Maia','Merope''Alcyone','Atlas','Aldebaran','Capella','Elnath','Menkalinan','Alfirk','Alpheratz','Mirfak']

    for star_obj in star_objs:
        do_eval(star_obj[0], star_obj[1:], object_file)



    quit()

    full_data = object_file['xml_counts']
    star_name = 'Sirrah'
    star_x = 1818.761111278641
    star_y = 1335.8668986857397
    variation_val = 30      #maybe try with like iterative process to go out more and more
    star_alt = 0.8095796704292297
    star_az = 4.574984550476074
    cut_data = full_data[int(star_x) - variation_val:int(star_x) + variation_val, int(star_y) - variation_val:int(star_y) + variation_val]
    #cut_data is now our are where the star is in

    # cut_data_woback = background_noise_est(cut_data)
    # sources_detected = source_detection(cut_data_woback)

    # #this is our astropy table
    # #columns;
    # '''id, xcentroid, ycentroid, sharpness, roundness1, roundness2, npix, sky, peak, flux, mag'''

    # print(star_name)
    # print(list(sources_detected['flux']))
    # print(list(sources_detected['mag']))
    # print(list(sources_detected['roundness1']))
    # print(list(sources_detected['roundness2']))
    # print(list(sources_detected['sharpness']))
    # print(list(sources_detected['npix']))

    '''
    Sirrah
    flux            1.4988088
    mag             -0.43936562
    roundness1      0.45053891
    roundness2      0.20014221
    sharpness       0.54189533
    npix            25

    '''

    ########




    #Done
    t_fin = time.perf_counter()
    print('\n \n \n********* Process-{0} over ************'.format(pid))
    print('Runtime: {0} seconds'.format(t_fin - t_start))
    p_mem = process.memory_info().rss
    print('Memory used: {0} bytes ({1} GB) '.format(p_mem, p_mem/(1e9)))
