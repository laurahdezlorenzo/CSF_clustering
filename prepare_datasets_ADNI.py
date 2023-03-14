import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

def abeta_input(x):
    if x == '>1700':
        return random.randrange(1700, 2000)
    else:
        return x

def load_data_ADNI(input_filename):
    
#     # Get samples names
#     samples_file = open('data/ADNI/field_names.txt', 'r')
#     sample_ids = samples_file.read().split('\n')
#     samples_file.close()
    
#     sample_ids = [i.upper() for i in sample_ids]
#     del sample_ids[0:79]
#     print('\nFrom ADNIMERGE, samples in genetic cohort:', len(sample_ids))
    
    # Load original ADNIMERGE data
    adnimerge = pd.read_csv(input_filename, index_col='RID', low_memory=False)
    # adnimerge.index = adnimerge.index.str.upper()
    
    # Replace some string values
    adnimerge['ABETA'].replace('>1700', 1700, inplace=True)
    adnimerge['PTAU'].replace('<8', 8, inplace=True)
    adnimerge['PTAU'].replace('>120', 120, inplace=True)
    adnimerge['TAU'].replace('>1300', 1300, inplace=True)
    adnimerge['TAU'].replace('<80', 80, inplace=True)
    adnimerge['ABETA'].replace('<200', 200, inplace=True)
    adnimerge['ABETA'] = adnimerge['ABETA'].astype('float64')
    adnimerge['PTAU'] = adnimerge['PTAU'].astype('float64')
    adnimerge['TAU'] = adnimerge['TAU'].astype('float64')

    adnimerge['ABETA_bl'].replace('>1700', 1700, inplace=True)
    adnimerge['PTAU_bl'].replace('<8', 8, inplace=True)
    adnimerge['PTAU_bl'].replace('>120', 120, inplace=True)
    adnimerge['TAU_bl'].replace('>1300', 1300, inplace=True)
    adnimerge['TAU_bl'].replace('<80', 80, inplace=True)
    adnimerge['ABETA_bl'].replace('<200', 200, inplace=True)
    adnimerge['ABETA_bl'] = adnimerge['ABETA_bl'].astype('float64')
    adnimerge['PTAU_bl'] = adnimerge['PTAU_bl'].astype('float64')
    adnimerge['TAU_bl'] = adnimerge['TAU_bl'].astype('float64')


    # adnimerge['ABETA'] = adnimerge['ABETA'].apply(lambda x: abeta_input(x))
    
#     # Select information only from genetic cohort
#     adnimerge_genetics = adnimerge.loc[sample_ids]
    
#     # Save new dataset
#     adnimerge_genetics.to_csv('data/ADNIMERGE_genetics.csv')
    
    return adnimerge

def select_n_years_diagnosis(diagnosis_df, n_years):

    n_months = n_years*12

    samples_to_drop = []
    samples_5years = []
    samples_strange = []

    print(f'\tOriginal: {diagnosis_df.shape[0]}')

    counter = 0
    for index, row in diagnosis_df.iterrows():
        tmp = row[row.notna()]
        months = tmp.index.to_list()
        
        if months[-1] < n_months:
            samples_to_drop.append(index)
            samples_5years.append(index)
        
        first = tmp[months[0]]
        last = tmp[months[-1]]
        
        if first == 'MCI' and last == 'CN':
            samples_to_drop.append(index)
            samples_strange.append(index)
            
        if first == 'Dementia' and last == 'MCI':
            samples_to_drop.append(index)
            samples_strange.append(index)
        
        if first == 'Dementia' and last == 'CN':
            samples_to_drop.append(index)
            samples_strange.append(index)
        
        if first == 'Dementia' and last == 'Dementia':
            samples_to_drop.append(index)
            samples_strange.append(index)
    
    print(f'\t3 years diagnosis: {diagnosis_df.shape[0] - len(samples_5years)}')
    print(f'\tStrange diagnosis filters: {diagnosis_df.shape[0] - len(set(samples_to_drop))}')

    result = diagnosis_df.drop(samples_to_drop).index.to_list()

    return result

def select_min_points(df, b, threshold):

    timeseries = {'AV45': [0, 24, 48, 72, 96],
                    'ABETA': [0, 12, 24, 36, 48],
                    'TAU': [0, 12, 24, 36, 48],
                    'PTAU': [0, 12, 24, 36, 48],
                    'FDG': [0, 12, 24, 36, 48, 60],
                    'Ventricles': [0, 12, 24, 36, 48],
                    'Hippocampus': [0, 12, 24, 36, 48],
                    'WholeBrain': [0, 12, 24, 36, 48],
                    'Entorhinal': [0, 12, 24, 36, 48],
                    'Fusiform': [0, 12, 24, 36, 48],
                    'MidTemp': [0, 12, 24, 36, 48]}
    
    ts = timeseries[b]
    sel = df[ts]

    print('\tTimeseries selected:', ts)

    # print(df)
    # print(sel)
    
    samples_points = []
    for index, row in sel.iterrows():
        tmp = row[row.notna()]
        months = tmp.index.to_list()

        if len(months) >= threshold:
            samples_points.append(index)

    df_points = sel.loc[samples_points]
            
    return df_points

def get_dx_changes(diagnosis_df):

    dx_changes = {}
    dx_changes_months = {} # store month of last diagnosis
    dx_to_dementia = {}

    for index, row in diagnosis_df.iterrows():
        tmp = row[row.notna()]
        months = tmp.index.to_list()
        
        first = tmp[months[0]]
        last = tmp[months[-1]]
        # first = first_df.loc[index].values[0]
        # print(index, first)

        todem = tmp.eq('Dementia').idxmax()
        dxchange = f'{first} to {last}'

        if last != 'Dementia':
            # print(dxchange)
            todem = 0
        
        dx_changes[index] = dxchange
        dx_changes_months[index] = months[-1]
        dx_to_dementia[index] = todem
    
    return dx_changes, dx_changes_months, dx_to_dementia

def fill_baseline_missings(biomarker_df, dx_df, b):

    # Samples per baseline diagnosis
    cn = biomarker_df.loc[dx_df[0] == 'CN'].index.to_list()
    mci = biomarker_df.loc[dx_df[0] == 'MCI'].index.to_list()
    dem = biomarker_df.loc[dx_df[0] == 'Dementia'].index.to_list() # ninguno empieza en Dementia, bien

    print('Baseline missing values filling')
    print()
    
    print('\tCN with baseline visit:', len(cn))
    print('\tMCI with baseline visit:', len(mci))
    print('\tDementia with baseline visit:', len(dem))
    print()
    
    # Input baseline missing values with baseline means per diagnosis group
    mean_baseline_cn = biomarker_df.loc[cn][0].mean()
    mean_baseline_mci = biomarker_df.loc[mci][0].mean()
    mean_baseline_dem = biomarker_df.loc[dem][0].mean()
    
    cn_df = biomarker_df.loc[cn]
    mci_df = biomarker_df.loc[mci]
    dem_df = biomarker_df.loc[dem]
    
    cn_df[0].replace(np.nan, mean_baseline_cn, inplace=True)
    mci_df[0].replace(np.nan, mean_baseline_mci, inplace=True)
    dem_df[0].replace(np.nan, mean_baseline_dem, inplace=True)
    
    print('\tCN baseline:', round(mean_baseline_cn, 4))
    print('\tMCI baseline:', round(mean_baseline_mci, 4))
    print('\tDementia baseline:', round(mean_baseline_dem, 4))
    print()
    
    result = pd.concat([cn_df, mci_df, dem_df])
    
    return result

def process_descriptive_variables(df):

    '''Process diagnosis and sociodemographical data'''

    dx = df[['Month', 'DX']]
    dx = dx.sort_values(by = ['RID', 'Month'], ascending = [True, True])
    dx = dx.pivot_table(values='DX', index=dx.index, columns='Month', aggfunc='first')
    dx = dx.dropna(axis='rows', how='all')
    dx = dx.dropna(axis='columns', how='all')

    demo = df[['Month', 'PTID', 'AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4', 'MMSE_bl', 'DX', 'DX_bl', 'CDRSB_bl']]
    demo = demo.sort_values(by = ['RID', 'Month'], ascending = [True, True])

    # Select patients that last minimum 3 years in the study
    samples_3years = select_n_years_diagnosis(dx, 5)

    # Diagnosis and sociodemographical data
    dx_3years = dx.loc[samples_3years]
    demo_3years = demo.loc[samples_3years]

    # Get diagnosis transitions and other variables to label clusters
    dx_changes, dx_last_months, dxtodem = get_dx_changes(dx_3years)
    demo_label = demo_3years[~demo_3years.index.duplicated(keep='last')]
    demo_label = demo_label.drop(columns=['Month', 'DX'])
    demo_label['DXCHANGE'] = pd.Series(dx_changes)
    demo_label['DX_last_month'] = pd.Series(dx_last_months)
    demo_label['DXTODEM'] = pd.Series(dxtodem)

    return demo_label, dx_3years

def process_input_variables(b, df, dx_3years):

    '''Process selected biomarker data'''

    samples = dx_3years.index.to_list()

    bio = df[['Month', b]]
    bio = bio.sort_values(by = ['RID', 'Month'], ascending = [True, True])

    bio_3years = bio.loc[samples]
    bio_3years = bio_3years.reset_index()
    bio_3years = bio_3years.set_index(['RID', 'Month'])
    bio_3years = bio_3years[~bio_3years.index.duplicated(keep='first')]
    bio_3years = bio_3years.unstack('Month')
    bio_3years.columns = bio_3years.columns.droplevel()

    # Select patients that have a miminum number N not-NaN timepoints in the biomarker timeseries
    # in this case we're selecting timeseries with a minimum of 4 timepoints
    # bio_4points = select_min_points(bio_3years, b, 3)
    # print(f'\t3 points filtering: {bio_4points.shape[0]}')
    # print()

    # Missing values handling and timeseries transformation
    # bio_fill1 = fill_baseline_missings(bio_3years, dx_3years, b) # baseline visit filling
    bio_fill1 = bio_3years.T
    # bio_fill2  = bio_fill1.T.interpolate(method='linear', limit=bio_4points.shape[1]-1) # in-between filling; limit = number of datapoints in ts - 1

    return bio_fill1

def process_input_variables_multidim(bios):

    '''Process selected biomarker data - more dimensions'''

    indata1 = pd.read_csv(f'/home/laura/Documents/CODE/APP_genetics/timeseries_biomarkers/data/ADNI/{bios[0]}_data.csv', index_col=0)
    indata2 = pd.read_csv(f'/home/laura/Documents/CODE/APP_genetics/timeseries_biomarkers/data/ADNI/{bios[1]}_data.csv', index_col=0)
    indata3 = pd.read_csv(f'/home/laura/Documents/CODE/APP_genetics/timeseries_biomarkers/data/ADNI/{bios[2]}_data.csv', index_col=0)

    # con estas lineas que siguen me aseguro el mismo orden en los 3 dfs
    merge = pd.concat([indata1, indata2, indata3], join='inner', axis=1)
    samples_ts = merge.index.to_list()

    indata1 = indata1.loc[merge.index.to_list()]
    indata2 = indata2.loc[merge.index.to_list()]
    indata3 = indata3.loc[merge.index.to_list()]

    # creamos el array con las 3 modalidades

    m1 = indata1.to_numpy()
    m2 = indata2.to_numpy()
    m3 = indata3.to_numpy()

    n = len(m1) # numero de muestras

    m = [m1, m2, m3] # lista con las modalidades

    samples = [] # lista de muestras final
    for k in range(n): # para cada paciente
        timestamps = [] # lista timeseries final
        for j in range(5): # para cada timestamp
            modalities = [] # lista de modalidades final
            for i in range(len(m)): # para cada modalidad
                modalities.append(m[i][k][j]) # guardamos valor biomarcador i del tiempo j de la muestra k
            timestamps.append(modalities) # guardamos los valores de los biomarcadores de cada tiempo
        samples.append(timestamps) # guardamos los valores de los biomarcadores de cada muestra
    
    all_ts = np.array(samples) # convertimos la lista final en un array   

    return all_ts, samples_ts

if __name__ == "__main__":

    data = load_data_ADNI('/home/laura/Documents/CODE/APP_genetics/timeseries_biomarkers/data/ADNI/ADNIMERGE.csv')
    print()

    for biomarker in ['ABETA', 'TAU', 'PTAU']:

        print()
        print('Results for:', biomarker)
        print('----------------------------------------')

        descriptive_data, diagnosis_3years = process_descriptive_variables(data)
        biomarker_data = process_input_variables(biomarker, data, diagnosis_3years)
        biomarker_data.T.to_csv(f'da/home/laura/Documents/CODE/APP_genetics/timeseries_biomarkers/datata/ADNI/{biomarker}_data.csv')
        


    



    
                                            
