from asyncore import read
from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import corner
import seaborn as sns
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Serif"],
})

def read_bk(cap, comp, ell, tracer, real):
    '''
    cap: 'S' or 'N'
    comp: 'Complete', 'Incomplete', 'Corrected'
    ell: '000', '202', '220', '112', or '404'
    tracer: 'LRG', 'ELG' or 'QSO'
    real: realisation number
    '''

    bk_dir = './measurements/bk/%s' %(tracer)
    filename = os.path.join(bk_dir, 'bk%s_diag%s_%s_%s_%s.txt' % (ell, tracer, cap, real, comp))

    df = np.loadtxt(filename, skiprows=12, usecols=[0,5,7])
    df = pd.DataFrame(data=df, columns=['k1_cen', 'Re{bk%s_raw}' %ell, 'Re{bk%s_shot}' %ell])
    a, b, c = [], [], []
    a.append(df['k1_cen'].to_list())
    b.append(df['Re{bk%s_raw}' %ell].to_list())
    c.append(df['Re{bk%s_shot}' %ell].to_list())

    return df

def plot_bk(ell, tracer):
    '''
    Compare the corrected and contaminated results with the complete results.

    :param: ell: '000', '202', '220', '112', or '404'
    :param: tracer: 'LRG', 'ELG' or 'QSO'
    '''

    plt.clf()
    sns.set_palette(sns.color_palette('hls', 2))

    for cap in ['S', 'N']:
        # Lists for complete, corrected, and contaminated results.
        comp, corr, cont = [], [], []

        # Loop across the 25 realisations and append results 
        # to the correct list, subtracting shot noise.
        for i in range(0,25):
            complete, incomplete, corrected =\
                read_bk(cap, 'Complete', ell, tracer, i),\
                read_bk(cap, 'Incomplete', ell, tracer, i),\
                read_bk(cap, 'Corrected', ell, tracer, i)

            comp_bk = complete['Re{bk%s_raw}' %(ell)]-complete['Re{bk%s_shot}'%ell]
            corr_bk = corrected['Re{bk%s_raw}' %(ell)]-corrected['Re{bk%s_shot}'%ell]
            cont_bk = incomplete['Re{bk%s_raw}' %(ell)]-incomplete['Re{bk%s_shot}'%ell]

            comp.append(comp_bk)
            corr.append(corr_bk)
            cont.append(cont_bk)
    
        # Take the power spectrum mean for each k over the 25 realisations.
        comp_array = [np.array(x) for x in comp]
        comp_bk = np.array([np.mean(k) for k in zip(*comp_array)])

        corr_array = [np.array(x) for x in corr]
        corr_bk = np.array([np.mean(k) for k in zip(*corr_array)])

        cont_array = [np.array(x) for x in cont]
        cont_bk = np.array([np.mean(k) for k in zip(*cont_array)])

        # Calculate the standard deviation over the 25 realisations.
        std_dev = np.array([np.std(k) for k in zip(*corr_array)])

        plt.axhline(y = 0, color = 'b', linestyle = 'dashed')
    
        if cap == 'N':
            n_dev_corr = (comp_bk-corr_bk)/std_dev
            n_dev_cont = (comp_bk-cont_bk)/std_dev

            # Plot NGC power spectrum differences.
            ax = sns.lineplot(x=complete['k1_cen'], y=n_dev_corr, 
                              label="NGC $\mathrm{x}=$ corrected", color='g')
            
            ax = sns.lineplot(x=complete['k1_cen'], y=n_dev_cont, 
                              label="NGC $\mathrm{x}=$ contaminated", 
                              color='g', linestyle='dotted')

        elif cap == 'S':
            s_dev_corr = (comp_bk-corr_bk)/std_dev
            s_dev_cont = (comp_bk-cont_bk)/std_dev

            # Plot SGC power spectrum differences.
            ax = sns.lineplot(x=complete['k1_cen'], y=s_dev_corr, 
                              label="SGC $\mathrm{x}=$ corrected", color='r')
            ax = sns.lineplot(x=complete['k1_cen'], y=s_dev_cont, 
                              label="SGC $\mathrm{x}=$ contaminated", 
                              color='r', linestyle='dotted')
    
    # Set title and axis labels. 
    # ax.set_title("DESI Y1 %s Power Spectrum $\ell=%s$ (Triumvirate)" %(ell, comp))
    ax.set_xlabel("$k^2$ [$h$Mpc$^{-1}$]")
    ax.set_ylabel("$(B(k)_\mathrm{comp}-B(k)_\mathrm{x})/\sigma_\mathrm{corr}$")
    ax.set_xlim(0.01, 0.21)
    ax.set_ylim(-5, 30)
    plt.rcParams['figure.dpi'] = 800
    plt.legend(prop={'size': 8})

    plt.savefig('./%s_%s_bk_analysis' %(ell, tracer), dpi=1000)

    return n_dev_corr, n_dev_cont, s_dev_corr, s_dev_cont