import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from scipy.spatial import Voronoi, voronoi_plot_2d

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score

from prepare_datasets_HCSC import load_data_HCSC
from prepare_datasets_ADNI import load_data_ADNI, process_descriptive_variables


def kmeans(dataset, n, ds_name, scaling):

   x = dataset.dropna()

   if scaling == True:
        scaler = StandardScaler()
        x_std = scaler.fit_transform(x)

   elif scaling == False:
        x_std = x

   km = KMeans(n_clusters=n, max_iter=300, random_state=0)
   y_km = km.fit_predict(x_std)
   centroids = km.cluster_centers_

   if n == 3 and scaling == False:

      tmp = [centroids[0][0], centroids[1][0], centroids[2][0]]
      tmp.sort()

      c_01 = round((tmp[0] + tmp[1])/2, 4)
      c_12 = round((tmp[1] + tmp[2])/2, 4)

      print('New cut-off points:')
      print('  -', c_01, c_12)
   
   elif n == 2 and scaling == False:

      tmp = [centroids[0][0], centroids[1][0]]
      tmp.sort()

      c_01 = round((tmp[0] + tmp[1])/2, 4)

      print('New cut-off points:')
      print('  -', c_01)
        
   silhouette_avg = round(silhouette_score(x_std, y_km, metric='euclidean'), 4)

   result = x.copy()
   result[f'n{n}_{ds_name}'] = y_km
   result = result[f'n{n}_{ds_name}'] # obtain only cluster assignations

   return result, silhouette_avg

def hcsc_clustering(N):

   # HCSC dataset clustering
   # Load data
   infile = 'data/HCSC/BaseLCR_24julio22conATNparaLaura.sav'
   descriptive_vars, lcr_vars, cognitive_vars = load_data_HCSC(infile)

   outliers = [510673, 1175866, 1106441, 622301, 2378737]
   descriptive_vars = descriptive_vars.drop(outliers)
   lcr_vars = lcr_vars.drop(outliers)
   cognitive_vars = cognitive_vars.drop(outliers)

   # Experiments dictionary: different datasets to test clustering
   # Keys represent dataset name, first element is the df, second # clusters
   # The number of clusters was determined by elbow curve method (see ipynb)

   biomarkers = ['Amil1_42R', 'TAUtotal', 'pTAU', 'RatioR']
   
   results = []
   si_scores = []

   # CSF biomarkers
   for biomarker in biomarkers:

      print(biomarker)
      
      # Tabular format dataset
      dataset = lcr_vars[[biomarker]]
      dataset = dataset.dropna()
      
      # Clustering
      tmp_res, si = kmeans(dataset, N, biomarker, False)
      results.append(tmp_res)
      si_scores.append(si)
      print()

   # Clustering all
   print('ALL_ab42')
   tmp_res, si = kmeans(lcr_vars[['Amil1_42R', 'TAUtotal', 'pTAU']], N, 'ALL_ab42', True)
   results.append(tmp_res)
   si_scores.append(si)
   print()

   print('ALL_ratio')
   tmp_res, si = kmeans(lcr_vars[['RatioR', 'TAUtotal', 'pTAU']], N, 'ALL_ratio', True)
   results.append(tmp_res)
   si_scores.append(si)
   print()

   return si_scores, results

def adni_clustering(N):

   # Load data
   abeta = pd.read_csv(f'data/ADNI/ABETA_data.csv', index_col='RID').add_prefix('abeta_')
   ttau = pd.read_csv(f'data/ADNI/TAU_data.csv', index_col='RID').add_prefix('ttau_')
   ptau = pd.read_csv(f'data/ADNI/PTAU_data.csv', index_col='RID').add_prefix('ptau_')

   # abeta = abeta.loc[abeta['abeta_0'] < 1700]

   dataset_all = pd.concat([abeta, ttau, ptau], axis=1)
   dataset_all = dataset_all[['abeta_0', 'ptau_0', 'ttau_0']].dropna()

   # Delete saturated values (optional)
   # dataset_all = dataset_all.drop(dataset_all[dataset_all['abeta_0'] == 1700].index)

   index = dataset_all.index
   biomarkers  = ['ABETA', 'TAU', 'PTAU']

   # Single CSF biomarkers
   results = []
   si_scores = []

   for biomarker in biomarkers:

      print(biomarker)
      
      # Create tabular format dataset
      raw = pd.read_csv(f'data/ADNI/{biomarker}_data.csv', index_col=0)
      dataset = raw[['0']] # get only baseline visit
      dataset = dataset.dropna()
      dataset = dataset.loc[index] # get common patients

      
      # Clustering
      tmp_res, si = kmeans(dataset, N, biomarker, False)
      results.append(tmp_res)
      si_scores.append(si)
      print()
   

   # Clustering all
   print('ALL')
   tmp_res, si = kmeans(dataset_all, N, 'ALL', True)
   results.append(tmp_res)
   si_scores.append(si)
   print()

   return si_scores, results


if __name__ == "__main__":

   sns.set(font="Arial", font_scale=1.)
   sns.set_style('whitegrid')
   # sns.set_palette('hls')

   # scores_hcsc = pd.DataFrame(index=['AB42', 't-tau', 'p-tau', 'AB42/AB40 ratio', 'All_AB42', 'All_Ratio'])
   # scores_adni = pd.DataFrame(index=['AB42', 't-tau', 'p-tau', 'All_AB42'])

   # results_hcsc = []
   # results_adni = []

   # for N in range(2, 11):

   #    print(N)

   #    n_scores_hcsc, n_results_hcsc = hcsc_clustering(N)
   #    n_scores_adni, n_results_adni = adni_clustering(N)

   #    scores_hcsc[N] = n_scores_hcsc
   #    scores_adni[N] = n_scores_adni

   #    n_results_hcsc_df = pd.concat(n_results_hcsc, axis=1)
   #    n_results_adni_df = pd.concat(n_results_adni, axis=1)

   #    results_hcsc.append(n_results_hcsc_df)
   #    results_adni.append(n_results_adni_df)

   # results_hcsc_df = pd.concat(results_hcsc, axis=1, join='inner')
   # results_adni_df = pd.concat(results_adni, axis=1, join='inner')

   # print(results_hcsc_df)
   # print(results_adni_df)

   # results_hcsc_df.to_csv('results/clustering_HCSC.csv')
   # results_adni_df.to_csv('results/clustering_ADNI.csv')

   # scores_hcsc = scores_hcsc.stack().reset_index().rename(columns={'level_0':'biomarker', 'level_1':'clustering', 0:'SI'})
   # scores_adni = scores_adni.stack().reset_index().rename(columns={'level_0':'biomarker', 'level_1':'clustering', 0:'SI'})

   # scores_hcsc.to_csv('results/SI_scores_HCSC.csv')
   # scores_adni.to_csv('results/SI_scores_ADNI.csv')


   #############################################################################
   # SI scores plots
   scores_hcsc = pd.read_csv('results/SI_scores_HCSC.csv')
   scores_adni = pd.read_csv('results/SI_scores_ADNI.csv')

   sns.set(font="Arial", font_scale=1.)
   sns.set_style('white')
   sns.set_palette('hls', 3)

   modes = scores_hcsc['biomarker'].unique()
   # colors = sns.color_palette('hls', 6)
   # palette = {mode: color for mode, color in zip(modes, colors)}
   # palette = sns.color_palette('hls', 3)

   scores_hcsc = scores_hcsc.loc[(scores_hcsc['biomarker'] == 'All_AB42') | (scores_hcsc['biomarker'] == 'All_Ratio')]
   scores_adni = scores_adni.loc[(scores_adni['biomarker'] == 'All_AB42')]

   scores_hcsc['biomarker'] = scores_hcsc['biomarker'] + '_HCSC'
   scores_adni['biomarker'] = scores_adni['biomarker'] + '_ADNI'

   scores = pd.concat([scores_hcsc, scores_adni])

   fig = plt.figure(figsize=(5, 4))
   g = sns.lineplot(data=scores, y='SI', hue='biomarker', x='clustering', marker='o')
   g.legend().get_frame().set_edgecolor('w')
   plt.ylim(0.3, 0.6)
   plt.xlabel('Number of clusters')
   plt.ylabel('Silhouette Index')
   plt.xticks(range(2, 11))

   plt.savefig('figures/si_scores.png', dpi=300, bbox_inches='tight')
   plt.show()

   # handles, labels = plt.get_legend_handles_labels()
   # labels = ['All with Aβ(1-42) HCSC', 'All with Ratio HCSC', 'All with Ratio ADNI']
   # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize=12)

   # plt.show()

   # fig, (ax1, ax2)= plt.subplots(1, 2, figsize=(10, 4))

   # g1 = sns.lineplot(data=scores_hcsc, y='SI', hue='biomarker', x='clustering',
   #                marker='o', ax=ax1, palette=palette)
   # g1.legend().get_frame().set_edgecolor('w')
   # ax1.set_ylim(0.3, 0.8)
   # ax1.set_xlabel('Number of clusters')
   # ax1.set_ylabel('Silhouette Index')
   # ax1.set_title('(a) HCSC dataset')
   # ax1.set(xticks=range(2, 11))

   # g2 = sns.lineplot(data=scores_adni, y='SI', hue='biomarker', x='clustering',
   #                marker='o', ax=ax2, palette=palette)
   # g2.legend().get_frame().set_edgecolor('w')
   # ax2.set_ylim(0.3, 0.8)
   # ax2.set_xlabel('Number of clusters')
   # ax2.set_ylabel('Silhouette Index')
   # ax2.set_title('(b) ADNI dataset')
   # ax2.set(xticks=range(2, 11))

   # handles, labels = ax1.get_legend_handles_labels()
   # labels = ['Aβ(1-42)', 'tTau', 'pTau', 'Aβ(1-42)/Aβ(1-40) ratio', 'All with Aβ(1-42)', 'All with ratio']
   # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize=12)

   # ax1.get_legend().remove()
   # ax2.get_legend().remove()

   # plt.savefig('figures/si_scores.png', dpi=300, bbox_inches='tight')
   # plt.show()

