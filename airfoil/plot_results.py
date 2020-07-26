""" 
Plot results

Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib
import seaborn as sns
import pandas as pd
from scipy.spatial import ConvexHull

from run_experiment import read_config
from bezier_gan import BezierGAN
from simulation import evaluate
from evaluation import diversity_score
from shape_plot import plot_shape


def convert_perf(perf):
    perf[:,1] = perf[:,0]/perf[:,1]
    ind = np.logical_or(np.isinf(perf), np.isnan(perf))
    perf[ind] = 0
    return perf

def select_best_ind(perf, n_selected):
    sorted_ind = np.argsort(perf, axis=0)
    selected_ind = sorted_ind[-n_selected:]
    return selected_ind
    
def plot_airfoils(airfoils, perfs, ax):
    n = airfoils.shape[0]
    zs = np.vstack((np.zeros(n), 0.5*np.arange(n))).T
    for i in range(n):
        plot_shape(airfoils[i], zs[i, 0], zs[i, 1], ax, 1., False, None, c='k', lw=1.2, alpha=.7)
        plt.annotate('{:.2f}'.format(perfs[i]), xy=(zs[i, 0], zs[i, 1]+0.2), size=14)
    ax.axis('off')
    ax.axis('equal')
    

if __name__ == "__main__":
    
    config_fname = 'config.ini'
    list_models = ['GAN', 'MO-PaDGAN']
    
    ###############################################################################
    # Plot diversity and quality scores
    print('Plotting diversity and quality scores ...')
    plt.rcParams.update({'font.size': 14})
    
    n = 1000 # generated sample size for each trained model
    subset_size = 10 # for computing DDP
    sample_times = n # for computing DDP
    
    # Training data
    x_path = './data/xs_train.npy'
    y_path = './data/ys_train.npy'
    airfoils_data = np.load(x_path)
    ind = np.random.choice(airfoils_data.shape[0], n, replace=False)
    airfoils_data = airfoils_data[ind]
    div_data = diversity_score(airfoils_data, subset_size, sample_times)
    qa_data = np.load(y_path)[ind]
    qa_data = convert_perf(qa_data)
    
    list_div = [div_data]
    list_qa0 = [qa_data[:,0]]
    list_qa1 = [qa_data[:,1]]
    
    for model_name in list_models:
        
        if model_name == 'GAN':
            lambda0, lambda1 = 0., 0.
        elif model_name == 'MO-PaDGAN':
            _, _, _, _, _, _, _, lambda0, lambda1, _ = read_config(config_fname)
            
        save_dir = './trained_gan/{}_{}'.format(lambda0, lambda1)
        airfoils = np.load('{}/gen_xs.npy'.format(save_dir))[:n]
        div = diversity_score(airfoils, subset_size, sample_times)
        qa = np.load('{}/gen_ys.npy'.format(save_dir))[:n]
        qa = convert_perf(qa)
            
        list_div.append(div)
        list_qa0.append(qa[:,0])
        list_qa1.append(qa[:,1])
    
    list_xlabels = ['Data',] + list_models
    
    fig = plt.figure(figsize=(9, 3))
    ax1 = fig.add_subplot(131)
    ax1.set_title('Diversity')
    ax1.boxplot(list_div, 0, '')
    ax1.set_xlim(0.5, len(list_xlabels) + 0.5)
    ax1.set_xticklabels(list_xlabels, rotation=10)
    ax2 = fig.add_subplot(132)
    ax2.set_title(r'$C_L$')
    ax2.boxplot(list_qa0, 0, '')
    ax2.set_xlim(0.5, len(list_xlabels) + 0.5)
    ax2.set_xticklabels(list_xlabels, rotation=10)
    ax3 = fig.add_subplot(133)
    ax3.set_title(r'$C_L/C_D$')
    ax3.boxplot(list_qa1, 0, '')
    ax3.set_xlim(0.5, len(list_xlabels) + 0.5)
    ax3.set_xticklabels(list_xlabels, rotation=10)
    plt.tight_layout()
    plt.savefig('./trained_gan/airfoil_scores.svg')
    plt.savefig('./trained_gan/airfoil_scores.pdf')
    plt.savefig('./trained_gan/airfoil_scores.png')
    plt.close()
    
    ###############################################################################
    # Plot airfoils in the performance space
    print('Plotting airfoils in the performance space ...')
    plt.rcParams.update({'font.size': 14})
    
    n = 100
    n_selected_airfoils = 5
    
    # Data
    x_path = './data/xs_train.npy'
    y_path = './data/ys_train.npy'
    airfoils_data = np.load(x_path)
    ind = np.random.choice(airfoils_data.shape[0], n, replace=False)
    airfoils_data = airfoils_data[ind]
    perfs_data = np.load(y_path)[ind]
    perfs_data = convert_perf(perfs_data)
    selected_ind_data = select_best_ind(perfs_data, n_selected_airfoils)
        
    # GAN
    x_path = './trained_gan/0.0_0.0/gen_xs.npy'
    y_path = './trained_gan/0.0_0.0/gen_ys.npy'
    airfoils_gan = np.load(x_path)[:n]
    perfs_gan = np.load(y_path)[:n]
    perfs_gan = convert_perf(perfs_gan)
    selected_ind_gan = select_best_ind(perfs_gan, n_selected_airfoils)
        
    # MO-PaDGAN
    _, _, _, _, _, _, _, lambda0, lambda1, _ = read_config(config_fname)
    x_path = './trained_gan/{}_{}/gen_xs.npy'.format(lambda0, lambda1)
    y_path = './trained_gan/{}_{}/gen_ys.npy'.format(lambda0, lambda1)
    airfoils_padgan = np.load(x_path)[:n]
    perfs_padgan = np.load(y_path)[:n]
    perfs_padgan = convert_perf(perfs_padgan)
    selected_ind_padgan = select_best_ind(perfs_padgan, n_selected_airfoils)
    
    perfs = np.concatenate((perfs_data, perfs_gan, perfs_padgan), axis=0)
    min_perf = np.min(perfs, axis=0)
    max_perf = np.max(perfs, axis=0)
    ranges = max_perf - min_perf
    min_perf -= ranges*.1
    max_perf += ranges*.1
    bounds = np.vstack((min_perf, max_perf))
        
    fig = plt.figure(figsize=(8, 4))
    
    ax1 = fig.add_subplot(121)
    # ax1.scatter(perfs_data[:,0], perfs_data[:,1], s=20, marker='o', edgecolors='none', c='#BCC6CC', label='Data')
    # ax1.scatter(perfs_gan[:,0], perfs_gan[:,1], s=20, marker='^', edgecolors='none', c='k', label='GAN')
    ax1.scatter(perfs_data[:,0], perfs_data[:,1], s=20, marker='o', edgecolors='none', c='#ffa600', label='Data')
    ax1.scatter(perfs_gan[:,0], perfs_gan[:,1], s=20, marker='^', edgecolors='none', c='#003f5c', label='GAN')
    ax1.legend(frameon=True)
    ax1.set_xlim(bounds[:,0])
    ax1.set_ylim(bounds[:,1])
    ax1.set_title('(a) GAN')
    ax1.set_xlabel(r'$C_L$')
    ax1.set_ylabel(r'$C_L/C_D$')
    
    ax2 = fig.add_subplot(122)
    # ax2.scatter(perfs_data[:,0], perfs_data[:,1], s=20, marker='o', edgecolors='none', c='#BCC6CC', label='Data')
    # ax2.scatter(perfs_padgan[:,0], perfs_padgan[:,1], s=20, marker='s', edgecolors='none', c='k', label='MO-PaDGAN')
    ax2.scatter(perfs_data[:,0], perfs_data[:,1], s=20, marker='o', edgecolors='none', c='#ffa600', label='Data')
    ax2.scatter(perfs_padgan[:,0], perfs_padgan[:,1], s=20, marker='s', edgecolors='none', c='#003f5c', label='MO-PaDGAN')
    ax2.legend(frameon=True)
    ax2.set_xlim(bounds[:,0])
    ax2.set_ylim(bounds[:,1])
    ax2.set_title('(b) MO-PaDGAN')
    ax2.set_xlabel(r'$C_L$')
    ax2.set_ylabel(r'$C_L/C_D$')
    
    plt.tight_layout()
    plt.savefig('./trained_gan/airfoils_perf.svg')
    plt.savefig('./trained_gan/airfoils_perf.pdf')
    plt.savefig('./trained_gan/airfoils_perf.png')
    plt.close()
    
    ###############################################################################
    # Plot top-ranked airfoils
    fig = plt.figure(figsize=(9, 3))
    ax1 = fig.add_subplot(161)
    plot_airfoils(airfoils_data[selected_ind_data[:,0]], perfs_data[selected_ind_data[:,0],0], ax1)
    ax1.set_title(r'$C_L$')
    ax2 = fig.add_subplot(162)
    plot_airfoils(airfoils_data[selected_ind_data[:,1]], perfs_data[selected_ind_data[:,1],1], ax2)
    ax2.set_title(r'$C_L/C_D$')
    ax3 = fig.add_subplot(163)
    plot_airfoils(airfoils_gan[selected_ind_gan[:,0]], perfs_gan[selected_ind_gan[:,0],0], ax3)
    ax3.set_title(r'$C_L$')
    ax4 = fig.add_subplot(164)
    plot_airfoils(airfoils_gan[selected_ind_gan[:,1]], perfs_gan[selected_ind_gan[:,1],1], ax4)
    ax4.set_title(r'$C_L/C_D$')
    ax5 = fig.add_subplot(165)
    plot_airfoils(airfoils_padgan[selected_ind_padgan[:,0]], perfs_padgan[selected_ind_padgan[:,0],0], ax5)
    ax5.set_title(r'$C_L$')
    ax6 = fig.add_subplot(166)
    plot_airfoils(airfoils_padgan[selected_ind_padgan[:,1]], perfs_padgan[selected_ind_padgan[:,1],1], ax6)
    ax6.set_title(r'$C_L/C_D$')
    plt.tight_layout()
    plt.savefig('./trained_gan/top_airfoils.svg')
    plt.savefig('./trained_gan/top_airfoils.pdf')
    plt.savefig('./trained_gan/top_airfoils.png')
    plt.close()
    
    ###############################################################################
    # Plot airfoils embedding
    print('Plotting airfoils embedding ...')
    plt.rcParams.update({'font.size': 14})
            
    xs = np.concatenate([airfoils_data, airfoils_gan, airfoils_padgan], axis=0)
    xs = xs.reshape(xs.shape[0], -1)
    scaler_x = MinMaxScaler()
    xs = scaler_x.fit_transform(xs)
    tsne = TSNE(n_components=2)
    # tsne = PCA(n_components=2)
    zs = tsne.fit_transform(xs)
    scaler_z = MinMaxScaler()
    zs = scaler_z.fit_transform(zs)
        
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax1.scatter(zs[:n,0], zs[:n,1], s=20, marker='o', edgecolors='none', c='#ffa600', label='Data')
    ax1.scatter(zs[n:2*n,0], zs[n:2*n,1], s=20, marker='^', edgecolors='none', c='#003f5c', label='GAN')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are off
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=False,
        labelbottom=False)
    ax1.set_xlim([-.01, 1.01])
    ax1.set_ylim([-.05, 1.05])
    ax1.set_title('(a) GAN')
    
    ax2 = fig.add_subplot(122)
    ax2.scatter(zs[:n,0], zs[:n,1], s=20, marker='o', edgecolors='none', c='#ffa600', label='Data')
    ax2.scatter(zs[2*n:,0], zs[2*n:,1], s=20, marker='s', edgecolors='none', c='#003f5c', label='MO-PaDGAN')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are off
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=False,
        labelbottom=False)
    ax2.set_xlim([-.01, 1.01])
    ax2.set_ylim([-.05, 1.05])
    ax2.set_title('(b) MO-PaDGAN')
    
    plt.savefig('./trained_gan/airfoils_tsne.svg')
    plt.savefig('./trained_gan/airfoils_tsne.pdf')
    plt.savefig('./trained_gan/airfoils_tsne.png')
    plt.close()
    
    # ###############################################################################
    # # Compute volume of the convex hull
    # print('Computing convex hull volumes ...')
    # n = 500
    
    # # Training data
    # xs = np.load('./data/xs_train.npy')
    # ind = np.random.choice(xs.shape[0], n, replace=False)
    # xs = xs[ind]
    # hull = ConvexHull(xs.reshape(xs.shape[0], -1))
    # vol_data = hull.volume
    # print('Data: {}'.format(vol_data))
    
    # # GAN
    # xs = np.load('./trained_gan/0.0_0.0/gen_xs.npy')[:n]
    # hull = ConvexHull(xs.reshape(xs.shape[0], -1))
    # vol_gan = hull.volume
    # print('GAN: {}'.format(vol_gan))
    
    # # MO-PaDGAN
    # _, _, _, _, _, _, _, lambda0, lambda1, _ = read_config(config_fname)
    # xs = np.load('./trained_gan/{}_{}/gen_xs.npy'.format(lambda0, lambda1))[:n]
    # hull = ConvexHull(xs.reshape(xs.shape[0], -1))
    # vol_padgan = hull.volume
    # print('MO-PaDGAN: {}'.format(vol_padgan))
    
    