import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF

# plot anchor_p_filtered and simulated_anchor_p
def compare_real_simu_anchor_p(df, out_folder):
    df_to_plot = df[df.abundant_target == True]
    plt.scatter(x = df_to_plot.anchor_p_simulated, y = df_to_plot.anchor_p_filtered,s=5)
    plt.xlabel("anchor_p_simulated")
    plt.ylabel("anchor_p_filtered")
    plt.plot([0,1], [0,1], 'r', linewidth=1)
    plt.savefig(f"{out_folder}/compare_real_simu.png")
    plt.clf()
    
# plot anchor_p_filtered and anchor_p_unfiltered
def compare_anchor_p(df, out_folder):
    df_to_plot = df[df.abundant_target == True]
    plt.scatter(x = df_to_plot.anchor_p_unfiltered, y = df_to_plot.anchor_p_filtered,s=5)
    plt.xlabel("anchor_p_unfiltered")
    plt.ylabel("anchor_p_filtered")
    plt.plot([0,1], [0,1], 'r', linewidth=1)
    plt.savefig(f"{out_folder}/compare_anchor_filtered.png")
    plt.clf()

    
# plot ECDF
# def plot_ecdf(df):
#     """
#     A basic function that plots ECDF. Take in a merged df 
#     that contains both real and simulated target data. 
#     """
#     # drop duplicates and NAN
#     df_to_plot = df[['anchor', 'anchor_p_filtered', 'simu_anchor_p_filtered']].drop_duplicates().dropna()
    
#     if len(df_to_plot) == 0:
#         plt.ylim([-0.005,0.105])
#     else: 
#         # real data ecdf
#         ecdf = ECDF(df_to_plot["anchor_p_filtered"])
#         df_to_plot['ECDF'] = ecdf(df_to_plot["anchor_p_filtered"])
#         df_to_plot_sig = df_to_plot[df_to_plot['anchor_p_filtered'] < .1]
#         # simulated data ecdf
#         ecdf = ECDF(df_to_plot["simu_anchor_p_filtered"])
#         df_to_plot['simu_ECDF'] = ecdf(df_to_plot["simu_anchor_p_filtered"])
#         df_to_plot_sig_simu = df_to_plot[df_to_plot['simu_anchor_p_filtered'] < .1]
        
#         # plot
#         plt.scatter(df_to_plot_sig['anchor_p_filtered'], df_to_plot_sig['ECDF'], label='real_target', s=15, marker = 'o', facecolors='none', edgecolors='r')
#         plt.scatter(df_to_plot_sig_simu['simu_anchor_p_filtered'], df_to_plot_sig_simu['simu_ECDF'], label='simulated_target', s=15, marker = 'o', facecolors='none', edgecolors='b')

# plot ECDF (using simulation method 2)
def plot_ecdf(df):

    # drop duplicates and NAN
    df_to_plot = df[['anchor', 'anchor_p_BH']].drop_duplicates().dropna()
    
    if len(df_to_plot) == 0:
        plt.ylim([-0.005,0.105])
    else: 
        # real data ecdf
        ecdf = ECDF(df_to_plot["anchor_p_BH"])
        df_to_plot['ECDF'] = ecdf(df_to_plot["anchor_p_BH"])
#         df_to_plot_sig = df_to_plot[df_to_plot['anchor_p_BH'] < .1]

        # simulated data ecdf
        # ecdf = ECDF(df_to_plot["anchor_p_simulated"])
        # df_to_plot['simu_ECDF'] = ecdf(df_to_plot["anchor_p_simulated"])
        # df_to_plot_sig_simu = df_to_plot[df_to_plot['anchor_p_simulated'] < .1]

        # plot
        plt.plot(df_to_plot['anchor_p_BH'], df_to_plot['ECDF'], label='real_target', color='r')
        # plt.scatter(df_to_plot['anchor_p_simulated'], df_to_plot['simu_ECDF'], label='simulated_target', s=15, marker = 'o', facecolors='none', edgecolors='b')
        
        
# plot ecdf wihtout stratifying anchor_p
def ecdf_wrap_all_mut(df, DATA_HANDLE, out_folder):
    """
    A wrapper function that plots ECDF without 
    stratifying data by number of mutations.
    """
    plot_ecdf(df)
    plt.axvline(x = .05, color = 'g', linestyle='--', label ='p = .05')
#     plt.xlim([-0.005,0.105])
    _, ymax = plt.ylim()
    plt.plot([0,min(1,ymax)], [0,min(1,ymax)], 'k', linewidth=1, label = 'y = x')
    plt.xlabel('anchor_p')
    plt.ylabel('ECDF')
    plt.legend()
    plt.title(f"{DATA_HANDLE}_no_stratifying.png")
    plt.savefig(f"{out_folder}/{DATA_HANDLE}_ECDF_all.png")
    plt.clf()


# plot ecdf by stratifying anchor_p by number of mutaions in targets
def ecdf_wrap_sing_mut(df, DATA_HANDLE, out_folder):
    """
    A wrapper function that plots ECDF stratifying 
    data by number of mutations.
    """
    for totaMut in df['totaMut'].unique():
        df_totaMut = df[df['totaMut'] == totaMut]
        plot_ecdf(df_totaMut)
#         plt.axvline(x = .05, color = 'g', linestyle='--', label ='p = .05')
#         plt.xlim([-0.005,0.105])
        _, ymax = plt.ylim()
        plt.plot([0,min(1,ymax)], [0,min(1,ymax)], 'k', linewidth=1, label = 'y = x')
        plt.xlabel('anchor_p')
        plt.ylabel('ECDF')
        plt.legend()
        plt.title(f"{DATA_HANDLE}_total_mutation_{totaMut}") 
        plt.savefig(f"{out_folder}/{DATA_HANDLE}_ECDF_total_mutation_{totaMut}.png")
        plt.clf()
        
