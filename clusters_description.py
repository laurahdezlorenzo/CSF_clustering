import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from prepare_datasets_HCSC import load_data_HCSC
from prepare_datasets_ADNI import load_data_ADNI, process_descriptive_variables

def label_A(row, var, thres):
    if row[var] < thres:
        return 'A+'
    elif row[var] >= thres:
        return 'A-'
    
def label_T(row, var, thres):
    if row[var] > thres:
        return 'T+'
    elif row[var] <= thres:
        return 'T-'
    
def label_N(row, var, thres):
    if row[var] > thres:
        return 'N+'
    elif row[var] <= thres:
        return 'N-'

def load_cluster_data_HCSC():

    clustering = pd.read_csv('results/clustering_HCSC.csv', index_col = 0)
    des_data, lcr_data, cog_data = load_data_HCSC('data/HCSC/BaseLCR_24julio22conATNparaLaura.sav')

    des_data['DxclinBreve'].replace({'EA GDS3':'LMCI', 'EApreMCI':'EMCI', 'Control/QSM':'SMC', 'Otros':'MCI-NN'}, inplace=True)

    # Replacements for comparing easily with ATN system
    clustering['n3_pTAU'].replace({0: 'A', 1:'B', 2:'C'}, inplace=True)
    clustering['n3_pTAU'].replace({'A': 2, 'B': 0, 'C':1}, inplace=True)
    # clustering['n3_ALL_ab42'].replace({0: 'A', 1:'B', 2:'C'}, inplace=True) # only for Supp Fig 1
    # clustering['n3_ALL_ab42'].replace({'A': 1, 'B': 2, 'C':0}, inplace=True)

    outliers = [510673, 1175866, 1106441]
    des_data = des_data.drop(outliers)
    lcr_data = lcr_data.drop(outliers)

    atn = lcr_data.copy()
    atn['A'] = atn.apply(lambda row:label_A(row, 'RatioR', 0.068), axis=1)
    atn['T'] = atn.apply(lambda row:label_T(row, 'pTAU', 59), axis=1)
    atn['N'] = atn.apply(lambda row:label_N(row, 'TAUtotal', 410), axis=1)
    atn['ATN'] = atn['A'] + atn['T'] + atn['N']

    a = pd.concat([des_data, lcr_data, atn['A'].rename('ATN_category'), atn['ATN'], clustering['n2_Amil1_42R'].rename('cluster')], axis=1)
    r = pd.concat([des_data, lcr_data, atn['A'].rename('ATN_category'), atn['ATN'], clustering['n2_RatioR'].rename('cluster')], axis=1)
    t = pd.concat([des_data, lcr_data, atn['T'].rename('ATN_category'), atn['ATN'], clustering['n3_pTAU'].rename('cluster')], axis=1)
    n = pd.concat([des_data, lcr_data, atn['N'].rename('ATN_category'), atn['ATN'], clustering['n2_TAUtotal'].rename('cluster')], axis=1)
    all = pd.concat([des_data, lcr_data, atn['ATN'].rename('ATN_category'),  atn['ATN'], clustering['n3_ALL_ratio'].rename('cluster')], axis=1)

    all_data = pd.concat([des_data, lcr_data, atn, clustering[['n2_Amil1_42R', 'n2_RatioR', 'n3_pTAU', 'n2_TAUtotal', 'n3_ALL_ratio']]], axis=1, join='inner')
    # all_data.to_csv('home/laura/Documents/CODE/APP_genetics/timeseries_biomarkers/results/HCSC_data_with_clusters.csv')

    print(all)

    return a, t, n, r, all

def load_cluster_data_ADNI():

    clustering = pd.read_csv('results/clustering_ADNI.csv', index_col = 0)

    # Replacements for comparing easily with ATN system
    clustering['n2_TAU'].replace({0: 'A', 1:'B'}, inplace=True)
    clustering['n2_TAU'].replace({'A': 1, 'B': 0}, inplace=True)
    clustering['n3_PTAU'].replace({0: 'A', 1:'B'}, inplace=True)
    clustering['n3_PTAU'].replace({'A': 1, 'B': 0}, inplace=True)
    clustering['n3_ALL'].replace({0: 'A', 1:'B', 2:'C'}, inplace=True)
    clustering['n3_ALL'].replace({'A': 0, 'B': 2, 'C':1}, inplace=True)
    
    adni = load_data_ADNI('data/ADNI/ADNIMERGE.csv')
    des_data, dx_3years = process_descriptive_variables(adni)
    des_data[['DX_first', 'DX_last']] = des_data['DXCHANGE'].str.split(' to ', 1, expand=True)

    abeta = pd.read_csv(f'data/ADNI/ABETA_data.csv', index_col='RID').add_prefix('abeta_')
    ttau = pd.read_csv(f'data/ADNI/TAU_data.csv', index_col='RID').add_prefix('ttau_')
    ptau = pd.read_csv(f'data/ADNI/PTAU_data.csv', index_col='RID').add_prefix('ptau_')
    lcr_data = pd.concat([abeta, ttau, ptau], axis=1)
    lcr_data = lcr_data[['abeta_0', 'ptau_0', 'ttau_0']].dropna()

    atn = lcr_data.copy()
    atn['A'] = atn.apply(lambda row:label_A(row, 'abeta_0', 980), axis=1)
    atn['T'] = atn.apply(lambda row:label_T(row, 'ptau_0', 24), axis=1)
    atn['N'] = atn.apply(lambda row:label_N(row, 'ttau_0', 266), axis=1)
    atn['ATN'] = atn['A'] + atn['T'] + atn['N']

    a = pd.concat([des_data, lcr_data, atn['A'].rename('ATN_category'), atn['ATN'], clustering['n2_ABETA'].rename('cluster')], axis=1, join='inner')
    t = pd.concat([des_data, lcr_data, atn['T'].rename('ATN_category'), atn['ATN'], clustering['n3_PTAU'].rename('cluster')], axis=1, join='inner')
    n = pd.concat([des_data, lcr_data, atn['N'].rename('ATN_category'), atn['ATN'], clustering['n2_TAU'].rename('cluster')], axis=1, join='inner')
    all = pd.concat([des_data, lcr_data, atn['ATN'].rename('ATN_category'), atn['ATN'], clustering['n3_ALL'].rename('cluster')], axis=1, join='inner')

    all_data = pd.concat([des_data, lcr_data, atn, clustering[['n2_ABETA', 'n3_PTAU', 'n2_TAU', 'n3_ALL']]], axis=1, join='inner')
    # all_data.to_csv('home/laura/Documents/CODE/APP_genetics/timeseries_biomarkers/results/ADNI_data_with_clusters.csv')

    return a, t, n, all

def plot_diagnosis_distributions(all_hcsc, all_adni):

    order = ['LMCI', 'EMCI', 'SMC', 'MCI-NN', 'CN']

    all_hcsc = all_hcsc.rename(columns={'DxclinBreve': 'diag'})
    all_adni = all_adni.rename(columns={'DX_bl': 'diag'})

    sns.set_palette('mako', len(order))

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    i = 0
    for df in [all_hcsc, all_adni]:
        x_var, y_var = "cluster", 'diag'
        df_grouped = df.groupby([x_var])[[y_var]].value_counts(normalize=True).unstack(y_var)

        if 'CN' not in df_grouped.columns:
            df_grouped['CN'] = 0.0

        if 'MCI-NN' not in df_grouped.columns:
            df_grouped['MCI-NN'] = 0.0

        df_grouped = df_grouped[order]
        print(df_grouped)
        print()
        df_grouped.plot.bar(stacked=True, ax=ax[i])
        ax[i].get_legend().remove()
        ax[i].set_xlabel('', fontsize=1)
        ax[i].set_ylabel('', fontsize=1)   
        i += 1

    ax[0].set_ylabel('Percentage (%) in cluster', fontsize=14)

    ax[0].set_title('(a) HCSC dataset')
    ax[1].set_title('(b) ADNI dataset')

    ax[0].set_xticklabels(['Cluster 0', 'Cluster 1', 'Cluster 2'], fontsize=12, rotation=0)
    ax[1].set_xticklabels(['Cluster 0', 'Cluster 1', 'Cluster 2'], fontsize=12, rotation=0)


    handles, labels = ax[0].get_legend_handles_labels()
    legend_order = [4, 3, 2, 1, 0]
    handles = [handles[idx] for idx in legend_order]
    labels  = [labels[idx] for idx in legend_order]
    leg = fig.legend(handles, labels, bbox_to_anchor=(1.12, 0.8), ncol=1, fontsize=14, title='Diagnosis:\n', title_fontsize=14)
    leg._legend_box.align = "left"

    plt.savefig(f'figures/diagnosis_distributions.png', dpi=300, bbox_inches='tight')



def plot_diagnosis_distributions_HCSC(a, t, n, r, all):

    # order = ['EA GDS3', 'EApreMCI', 'Control/QSM', 'Otros']
    # order = ['MCI AD', 'pre-MCI', 'SMC', 'MCI-NN']
    order = ['LMCI', 'EMCI', 'SMC', 'MCI-NN', 'CN']
    diag = "DxclinBreve"

    # order = ['EA GDS3', 'EApreMCI', 'APPlogopénica', 'Control/QSM',  'COVID', 'DCL no EA', 'Psiq/depresión', 'EM']
    # diag = "Dxclin"

    sns.set_palette('mako', len(order))

    # fig, (ax1, ax2) = plt.subplots(2, 5, figsize=(15, 8))
    fig, ax1 = plt.subplots(1, 5, figsize=(16, 4))

    i = 0
    for df in [a, t, n, r, all]:
        x_var, y_var = "cluster", diag
        df_grouped = df.groupby([x_var])[[y_var]].value_counts(normalize=True).unstack(y_var)
        df_grouped['CN'] = 0.0
        df_grouped = df_grouped[order]
        print(df_grouped)
        print()
        df_grouped.plot.bar(stacked=True, ax=ax1[i])
        ax1[i].get_legend().remove()
        ax1[i].set_xlabel('', fontsize=1)
        ax1[i].set_ylabel('', fontsize=1)   
        i += 1

    # ax1[0].set_ylabel('Percentage (%) in cluster', fontsize=14)

    # ax1[0].set_title('Aβ(1-42) (n = 2)')
    # ax1[1].set_title('pTau (n = 3)')
    # ax1[2].set_title('tTau (n = 2)')
    # ax1[3].set_title('Aβ(1-42)/Aβ(1-40) ratio (n = 2)')
    # ax1[4].set_title('All with ratio (n = 2)')

    # ax1[0].set_xticklabels(['Cluster 0', 'Cluster 1'], fontsize=12, rotation=0)
    # ax1[1].set_xticklabels(['Cluster 0', 'Cluster 1', 'Cluster 2'], fontsize=12, rotation=0)
    # ax1[2].set_xticklabels(['Cluster 0', 'Cluster 1'], fontsize=12, rotation=0)
    # ax1[3].set_xticklabels(['Cluster 0', 'Cluster 1'], fontsize=12, rotation=0)
    # ax1[4].set_xticklabels(['Cluster 0', 'Cluster 1', 'Cluster 2'], fontsize=12, rotation=0)

    # handles, labels = ax1[0].get_legend_handles_labels()
    # legend_order = [4, 3, 2, 1, 0]
    # handles = [handles[idx] for idx in legend_order]
    # labels  = [labels[idx] for idx in legend_order]
    # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(order), fontsize=12)
    plt.suptitle('(a) HCSC dataset', fontsize=14, x=0.16, y=1.02)
    plt.savefig(f'figures/{diag}_distributions_HCSC.png', dpi=300, bbox_inches='tight')

def plot_diagnosis_distributions_ADNI(a, t, n, all):

    order = ['LMCI', 'EMCI', 'SMC', 'MCI-NN', 'CN']
    diag  = "DX_bl"
    sns.set_palette('mako', len(order))

    fig, ax1 = plt.subplots(1, 5, figsize=(16, 4))

    i = 0
    for df in [a, t, n, all]:
        x_var, y_var = "cluster", diag
        df_grouped = df.groupby([x_var])[[y_var]].value_counts(normalize=True).unstack(y_var)
        df_grouped['MCI-NN'] = 0.0
        df_grouped = df_grouped[order]
        print(df_grouped)
        print()
        df_grouped.plot.bar(stacked=True, ax=ax1[i])
        ax1[i].get_legend().remove()
        ax1[i].set_xlabel('', fontsize=1)
        ax1[i].set_ylabel('', fontsize=1)   
        i += 1

    # ax1[0].set_ylabel('Percentage (%) in cluster', fontsize=14)

    # ax1[0].set_title('Aβ(1-42) (n = 2)')
    # ax1[1].set_title('pTau (n = 3)')
    # ax1[2].set_title('tTau (n = 2)')
    # ax1[3].set_title('All with Aβ(1-42) (n = 3)')

    # ax1[0].set_xticklabels(['Cluster 0', 'Cluster 1'], fontsize=12, rotation=0)
    # ax1[1].set_xticklabels(['Cluster 0', 'Cluster 1', 'Cluster 2'], fontsize=12, rotation=0)
    # ax1[2].set_xticklabels(['Cluster 0', 'Cluster 1'], fontsize=12, rotation=0)
    # ax1[3].set_xticklabels(['Cluster 0', 'Cluster 1', 'Cluster 2'], fontsize=12, rotation=0)

    # ax1[4].tick_params(axis='x', colors='white')
    # ax1[4].tick_params(axis='y', colors='white')
    # ax1[4].spines['left'].set_color('white')
    # ax1[4].spines['right'].set_color('white')
    # ax1[4].spines['bottom'].set_color('white')
    # ax1[4].spines['top'].set_color('white')   

    # handles, labels = ax1[0].get_legend_handles_labels()
    # legend_order = [4, 3, 2, 1, 0]
    # handles = [handles[idx] for idx in legend_order]
    # labels  = [labels[idx] for idx in legend_order]
    # title = 'Diagnosis:\n'   
    # leg = fig.legend(handles, labels, bbox_to_anchor=(0.89, 0.8), ncol=1, fontsize=14, title=title, title_fontsize=14)
    # leg._legend_box.align = "left"
    # plt.suptitle('(b) ADNI dataset', fontsize=14, x=0.16, y=1.02)
    # plt.savefig(f'figures/{diag}_distributions_ADNI.png', dpi=300, bbox_inches='tight')

def plot_ATN_distributions_HCSC(a, t, n, r, all):

    order = ['A-T-N-',    'A-T-N+',   'A-T+N-',   'A-T+N+',   'A+T-N-',   'A+T-N+',   'A+T+N-',   'A+T+N+']
    labels = ['A-T-(N-)', 'A-T-(N+)', 'A-T+(N-)', 'A-T+(N+)', 'A+T-(N-)', 'A+T-(N+)', 'A+T+(N-)', 'A+T+(N+)']

    i = 0
    for df in [a, t, n, r, all]:
        x_var, y_var = "cluster", "ATN"
        crosstab = pd.crosstab(df[y_var], df[x_var], normalize='columns')
        crosstab = round(crosstab.T, 4)*100
        crosstab['A-T+N-'] = 0.0
        crosstab = crosstab[order]

        print(crosstab)
        print()

        if i == 4:
            plt.figure(figsize=(6, 4))
            sns.heatmap(crosstab.T, annot=True, cmap="BuPu", vmin=0, vmax=100.0, fmt='.2f')
            plt.xticks([0.5, 1.5, 2.5], ['Cluster 0', 'Cluster 1', 'Cluster 2'], rotation=0)
            yticks = plt.yticks()
            plt.yticks(yticks[0], labels)
            plt.title('(a) HCSC dataset')
            plt.xlabel('', fontsize=0)
            plt.ylabel('', fontsize=0)
            plt.savefig(f'figures/ATN_distributions_HCSC.png', dpi=300, bbox_inches='tight')
        i +=1

def plot_ATN_distributions_ADNI(a, t, n, all):

    order = ['A-T-N-', 'A-T-N+', 'A-T+N-', 'A-T+N+', 'A+T-N-', 'A+T-N+', 'A+T+N-', 'A+T+N+']
    labels = ['A-T-(N-)', 'A-T-(N+)', 'A-T+(N-)', 'A-T+(N+)', 'A+T-(N-)', 'A+T-(N+)', 'A+T+(N-)', 'A+T+(N+)']

    i = 0
    for df in [a, t, n, all]:
        x_var, y_var = "cluster", "ATN"
        crosstab = pd.crosstab(df[y_var], df[x_var], normalize='columns')
        crosstab = round(crosstab.T, 4)*100
        crosstab['A+T-N+'] = 0.0
        crosstab = crosstab[order]

        print(crosstab)
        print()

        if i == 3:
            plt.figure(figsize=(6, 4))
            sns.heatmap(crosstab.T, annot=True, cmap="BuPu", vmin=0, vmax=100.0, fmt='.2f')
            plt.xticks([0.5, 1.5, 2.5], ['Cluster 0', 'Cluster 1', 'Cluster 2'], rotation=0)
            yticks = plt.yticks()
            plt.yticks(yticks[0], labels)
            plt.title('(b) ADNI dataset')
            plt.xlabel('', fontsize=0)
            plt.ylabel('', fontsize=0)
            plt.savefig(f'figures/ATN_distributions_ADNI.png', dpi=300, bbox_inches='tight')
        i +=1

def prepare_survival_datasets(df, k, atn):
        
        # Delete patients that already have dementia
        df = df[df['DX_bl'] != 'AD']
        
        followup = (df['DXTODEM']/12).rename('followup')
        event    = df['DX_last'].replace({'Dementia': True, 'CN': False, 'MCI': False}).rename('event')
        if atn == False:
            scenario = df['cluster'].rename('scenario')
        elif atn == True:
            scenario = df['ATN_category'].rename('scenario')
        lastdx   = df['DX_last_month'].rename('lastdx')
        gender   = df['PTGENDER'].replace({'Male': 0, 'Female': 1}).rename('sex')
        age      = df['AGE'].rename('age')
        educat   = df['PTEDUCAT'].rename('education')
        apoe4    = df['APOE4'].rename('apoe4')
        firstdx  = df['DX_bl'].rename('firstdx')

        survival = pd.concat([event, scenario, followup, firstdx, lastdx, gender, age, educat, apoe4],
                            axis=1, join='inner')

        survival.loc[survival['followup'] == 0, 'followup'] = survival['lastdx']/12
        survival['followup'] = np.round(np.where(survival['followup'] % 1 == 0.5,
                                    survival['followup'] + 0.1,
                                    survival['followup']))
        survival.loc[survival['followup'] >10, 'followup'] = 10
        
        if atn == False:
            survival.to_csv(f'data/ADNI/survival_plots/{k}_survival.csv')
        elif atn == True:
            survival.to_csv(f'data/ADNI/survival_plots/{k}_survival_ATN.csv')

def plot_scatterplots_HCSC(df):

    hue_order = [0, 1, 2]
    
    s = sns.color_palette('hls', 3)

    h = 'cluster'
    fig, axs = plt.subplots(1, 4, figsize=(22, 5))

    sns.scatterplot(data=df, x='Amil1_42R', y='RatioR', hue=h, hue_order=hue_order, ax=axs[3], palette=s)
    sns.scatterplot(data=df, x='Amil1_42R', y='pTAU', hue=h, hue_order=hue_order, ax=axs[0], palette=s)
    sns.scatterplot(data=df, x='Amil1_42R', y='TAUtotal', hue=h, hue_order=hue_order, ax=axs[1], palette=s)
    sns.scatterplot(data=df, x='pTAU', y='TAUtotal', hue=h, hue_order=hue_order, ax=axs[2], palette=s)

    axs[3].set_ylabel('Aβ(1-42)/Aβ(1-40) ratio', fontsize=14)
    axs[3].set_xlabel('Aβ(1-42)', fontsize=14)
    axs[0].set_xlabel('Aβ(1-42)', fontsize=14)
    axs[0].set_ylabel('pTau', fontsize=14)
    axs[1].set_xlabel('Aβ(1-42)', fontsize=14)
    axs[1].set_ylabel('tTau', fontsize=14)
    axs[2].set_xlabel('pTau', fontsize=14)
    axs[2].set_ylabel('tTau', fontsize=14)

    for ax in axs:
        ax.get_legend().remove()

    plt.suptitle('(a) HCSC dataset', fontsize=14, x=.16)
    plt.savefig('figures/scatterplots_HCSC.png', dpi=300, bbox_inches='tight')

def plot_scatterplots_ADNI(df):

    hue_order = [0, 1, 2]
    
    s = sns.color_palette('hls', 3)
    h = 'cluster'
    
    fig, axs = plt.subplots(1, 4, figsize=(22, 5))

    sns.scatterplot(data=df, x='abeta_0', y='ptau_0', hue=h, hue_order=hue_order, ax=axs[0], palette=s)
    sns.scatterplot(data=df, x='abeta_0', y='ttau_0', hue=h, hue_order=hue_order, ax=axs[1], palette=s)
    sns.scatterplot(data=df, x='ptau_0', y='ttau_0', hue=h, hue_order=hue_order, ax=axs[2], palette=s)

    axs[0].set_xlabel('Aβ(1-42)', fontsize=14)
    axs[0].set_ylabel('pTau', fontsize=14)
    axs[1].set_xlabel('Aβ(1-42)', fontsize=14)
    axs[1].set_ylabel('tTau', fontsize=14)
    axs[2].set_xlabel('pTau', fontsize=14)
    axs[2].set_ylabel('tTau', fontsize=14)

    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    axs[2].get_legend().remove()

    axs[3].tick_params(axis='x', colors='white')
    axs[3].tick_params(axis='y', colors='white')
    axs[3].spines['left'].set_color('white')
    axs[3].spines['right'].set_color('white')
    axs[3].spines['bottom'].set_color('white')
    axs[3].spines['top'].set_color('white')

    handles, labels = axs[0].get_legend_handles_labels()
    handles = [handles[idx] for idx in hue_order]
    labels  = ['Cluster 0', 'Cluster 1', 'Cluster 2']
    title = 'Clustering with all CSF biomarkers (n = 3):\n'    
    leg = fig.legend(handles, labels, bbox_to_anchor=(0.89, 0.8), ncol=1, fontsize=14, title=title, title_fontsize=14)
    leg._legend_box.align = "left"
    plt.suptitle('(b) ADNI dataset', x=0.16, fontsize=14)
    plt.savefig('figures/scatterplots_ADNI.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":

    sns.set(font="Arial", font_scale=1.)
    sns.set_style('white')

    a_hcsc, t_hcsc, n_hcsc, r_hcsc, all_hcsc = load_cluster_data_HCSC()
    a_adni, t_adni, n_adni, all_adni = load_cluster_data_ADNI()

    # plot_diagnosis_distributions_HCSC(a_hcsc, t_hcsc, n_hcsc, r_hcsc, all_hcsc)
    # plot_diagnosis_distributions_ADNI(a_adni, t_adni, n_adni, all_adni)

    plot_diagnosis_distributions(all_hcsc=all_hcsc, all_adni=all_adni)


    plot_ATN_distributions_HCSC(a_hcsc, t_hcsc, n_hcsc, r_hcsc, all_hcsc)
    print()
    print()
    plot_ATN_distributions_ADNI(a_adni, t_adni, n_adni, all_adni)

    plot_scatterplots_HCSC(all_hcsc)
    plot_scatterplots_ADNI(all_adni)

    # # Prepare survival datasets
    # datasets = {'ABETA': a_adni, 'PTAU': t_adni, 'TAU': n_adni, 'ALL':all_adni}
    # for biomarker in datasets:
    #     prepare_survival_datasets(datasets[biomarker], biomarker, False)

    # datasets = {'A': a_adni, 'T': t_adni, 'N': n_adni, 'ATN': all_adni}
    # for biomarker in datasets:
    #     prepare_survival_datasets(datasets[biomarker], biomarker, True)

    






