import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

def load_data_HCSC(spss_file):

    data = pd.read_spss(spss_file)

    #Change numeric identifiers to objects
    data['NHC'] = data['NHC'].astype('object')
    data['NHC_LCR'] = data['NHC_LCR'].astype('int').astype('object')
    data['Paciente_BaseLCR'] = data['Paciente_BaseLCR'].astype('int').astype('object')

    # set index to NHC_LCR bc LCR studies!
    data = data.set_index('NHC_LCR')

    des_file = open('/home/laura/Documents/CODE/APP_genetics/clustering_biomarkers/data/HCSC/variables/descriptive.txt', 'r')
    des = des_file.read().split('\n')
    des_file.close()

    lcr_file = open('/home/laura/Documents/CODE/APP_genetics/clustering_biomarkers/data/HCSC/variables/lcr.txt', 'r')
    lcr = lcr_file.read().split('\n')
    lcr_file.close()

    cog_file = open('/home/laura/Documents/CODE/APP_genetics/clustering_biomarkers/data/HCSC/variables/cognitive.txt', 'r')
    cog = cog_file.read().split('\n')
    cog_file.close()

    des_data = data[des].copy()
    lcr_data = data[lcr].copy()
    cog_data = data[cog].copy()

    lcr_data.loc[:,'RatioR'] = lcr_data['Amil1_42R']/lcr_data['Amil1_40']

    return des_data, lcr_data, cog_data

