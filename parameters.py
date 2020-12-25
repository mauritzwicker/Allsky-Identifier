##########################################
'''

author: @mauritzwicker
date: 01.12.2020
repo: star_detection_Allsky

Purpose: to show us the format to use for files

'''
##########################################

########
#IMPORTS
import time
import psutil
import os

#from other files

########




########
class Parameters_allsky:
    def __init__(self):
        #User defined parameters

        # Locations
        self.input_dir_data = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/outputs/data_out/'            #directory where input pickle file is
        self.image_full_PATH = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/outputs/data_out/20201118230658_eval.pickle'
        self.image_name = '20201118230658_eval.pickle'
        self.name_IMG = 'Star-Identification'
        self.dark_location = '/Volumes/Mauritz SeagateDrive/Mauritz/Anderes/AllSky_Daten/outputs/Darks/master_dark_allsky_10s.npy'

        # Observer Info
        self.observer_lat = '49.39'   #N
        self.observer_lat_letter = 'N'
        self.observer_long = '8.72'   #E
        self.observer_long_letter = 'E'
        self.observer_elevation = 560

        #for camera properties
        self.factors_ang_pix = 5.4  #5.4 arcmin / pixel
        self.awake_thresh = 30      #degree altitude for which stars above are considered awake
        self.deviation_position = 50 #diameter/2 of square around which pixel will still be considered
        self.x0_define = 1460
        self.y0_define = 1010

        #which stars to look for
        # self.stars_reference = ['Capella', 'Atlas', 'Betelgeuse', 'Rigel',
        #             'Aldebaran', 'Capella', 'Procyon', 'Polaris',
        #             'Dubhe', 'Mirach', 'Hamal', 'Almach', 'Deneb',
        #             'Vega', 'Regulus', 'Denebola', 'Alkaid', 'Spica',
        #             'Unukalhai', 'Rasalhague', 'Mirfak', 'Alpheratz',
        #             'Pollux', 'Altair', 'Diphda']

        self.stars_reference = ['Sirrah', 'Caph', 'Algenib', 'Schedar', 'Mirach', 'Achernar', 'Almach',
                                'Hamal', 'Polaris', 'Menkar', 'Algol', 'Electra', 'Taygeta', 'Maia',
                                'Merope', 'Alcyone', 'Atlas', 'Zaurak', 'Aldebaran', 'Rigel', 'Capella',
                                'Bellatrix', 'Elnath', 'Nihal', 'Mintaka', 'Arneb', 'Alnilam', 'Alnitak',
                                'Saiph', 'Betelgeuse', 'Menkalinan', 'Mirzam', 'Canopus', 'Alhena', 'Sirius',
                                'Adara', 'Wezen', 'Castor', 'Procyon', 'Pollux', 'Naos', 'Alphard', 'Regulus',
                                'Algieba', 'Merak', 'Dubhe', 'Denebola', 'Phecda', 'Minkar', 'Megrez',
                                'Gienah Corvi', 'Mimosa', 'Alioth', 'Vindemiatrix', 'Mizar', 'Spica',
                                'Alcor', 'Alcaid', 'Agena', 'Thuban', 'Arcturus', 'Izar', 'Kochab',
                                'Alphecca', 'Unukalhai', 'Antares', 'Rasalgethi', 'Shaula', 'Rasalhague',
                                'Cebalrai', 'Etamin', 'Kaus Australis', 'Vega', 'Sheliak', 'Nunki', 'Sulafat',
                                'Arkab Prior', 'Arkab Posterior', 'Rukbat', 'Albereo', 'Tarazed', 'Altair',
                                'Alshain', 'Sadr', 'Peacock', 'Deneb', 'Alderamin', 'Alfirk', 'Enif',
                                'Sadalmelik', 'Alnair', 'Fomalhaut', 'Scheat', 'Markab', 'Acamar', 'Acrux',
                                'Adhara', 'Alkaid', 'Alpheratz', 'Ankaa', 'Atria', 'Avior', 'Diphda',
                                'Eltanin', 'Formalhaut', 'Gacrux', 'Gienah', 'Hadar', 'Menkent', 'Miaplacidus',
                                'Mirfak', 'Rigil Kentaurus', 'Sabik', 'Suhail', 'Zubenelgenubi']

        # self.stars_reference = ['Mirach']

        #what to show
        self.plot_awake_stars = True    #whether to plot the stars found on image and show

########


