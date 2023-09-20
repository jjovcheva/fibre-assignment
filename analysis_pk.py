from asyncore import read
from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statistics import mean

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Serif"],
})

def read_pk(cap, comp, ell, tracer, real):
    '''
    Read in the power spectrum from a Triumvirate output file.

    cap: 'S' or 'N'
    comp: 'Complete', 'Incomplete', 'Corrected'
    ell: '0', '2' or '4'
    tracer: 'LRG', 'ELG' or 'QSO'
    real: realisation number
    '''

    # Point to the directory where measurements are saved.
    pk_dir = './measurements/pk/%s' %(tracer)
    filename = os.path.join(pk_dir, 'pk%s%s_%s_%s_%s.txt' % (ell, tracer, cap, real, comp))

    # Number of rows to skip may change with updates to Triumvirate.
    df = np.loadtxt(filename, skiprows=12, usecols=[0,3,5])
    df = pd.DataFrame(data=df, columns=['k_cen', 'Re{pk%s_raw}' %ell, 'Re{pk%s_shot}' %ell])
    a, b, c = [], [], []
    a.append(df['k_cen'].to_list())
    b.append(df['Re{pk%s_raw}' %ell].to_list())
    c.append(df['Re{pk%s_shot}' %ell].to_list())
    return df

def plot_pk(ell, tracer):
    '''
    Compare the corrected and contaminated results with the complete results.

    :param: ell: '0', '2' or '4'
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
                read_pk(cap, 'Complete', ell, tracer, i),\
                read_pk(cap, 'Incomplete', ell, tracer, i),\
                read_pk(cap, 'Corrected', ell, tracer, i)

            comp_pk = complete['Re{pk%s_raw}' %(ell)]-complete['Re{pk%s_shot}'%ell]
            corr_pk = corrected['Re{pk%s_raw}' %(ell)]-corrected['Re{pk%s_shot}'%ell]
            cont_pk = incomplete['Re{pk%s_raw}' %(ell)]-incomplete['Re{pk%s_shot}'%ell]

            comp.append(comp_pk)
            corr.append(corr_pk)
            cont.append(cont_pk)
    
        # Take the power spectrum mean for each k over the 25 realisations.
        comp_array = [np.array(x) for x in comp]
        comp_pk = np.array([np.mean(k) for k in zip(*comp_array)])

        corr_array = [np.array(x) for x in corr]
        corr_pk = np.array([np.mean(k) for k in zip(*corr_array)])

        cont_array = [np.array(x) for x in cont]
        cont_pk = np.array([np.mean(k) for k in zip(*cont_array)])

        # Calculate the standard deviation over the 25 realisations.
        std_dev = np.array([np.std(k) for k in zip(*corr_array)])

        plt.axhline(y = 0, color = 'b', linestyle = 'dashed')
    
        if cap == 'N':
            n_dev_corr = (comp_pk-corr_pk)/std_dev
            n_dev_cont = (comp_pk-cont_pk)/std_dev

            # Plot NGC power spectrum differences.
            ax = sns.lineplot(x=complete['k_cen'], y=n_dev_corr, 
                              label="NGC $\mathrm{x}=$ corrected", color='g')
            
            ax = sns.lineplot(x=complete['k_cen'], y=n_dev_cont, 
                              label="NGC $\mathrm{x}=$ contaminated", 
                              color='g', linestyle='dotted')

        elif cap == 'S':
            s_dev_corr = (comp_pk-corr_pk)/std_dev
            s_dev_cont = (comp_pk-cont_pk)/std_dev

            # Plot SGC power spectrum differences.
            ax = sns.lineplot(x=complete['k_cen'], y=s_dev_corr, 
                              label="SGC $\mathrm{x}=$ corrected", color='r')
            ax = sns.lineplot(x=complete['k_cen'], y=s_dev_cont, 
                              label="SGC $\mathrm{x}=$ contaminated", 
                              color='r', linestyle='dotted')
    
    # Set title and axis labels. 
    # ax.set_title("DESI Y1 %s Power Spectrum $\ell=%s$ (Triumvirate)" %(ell, comp))
    ax.set_xlabel("$k$ [$h$Mpc$^{-1}$]")
    ax.set_ylabel("$(P(k)_\mathrm{comp}-P(k)_\mathrm{x})/\sigma_\mathrm{corr}$")
    ax.set_xlim(0.01, 0.21)
    ax.set_ylim(-5, 30)
    plt.rcParams['figure.dpi'] = 800
    plt.legend(prop={'size': 8})

    plt.savefig('./%s_%s_pk_analysis' %(ell, tracer), dpi=1000)

    return n_dev_corr, n_dev_cont, s_dev_corr, s_dev_cont, complete

def analyse_pk(ell, tracer):

    n_dev_corr, n_dev_cont, s_dev_corr, s_dev_cont, k = plot_pk(ell, tracer)

    # Modify which statistic is being observed.
    chosen_stat = n_dev_cont
    
    minimum = min(chosen_stat)
    maximum = max(chosen_stat)

    k_spec = 0.071 # Pick out a specific wavenumber to look at.
    condition = np.where(k['k_cen']>k_spec, chosen_stat, 0)
    condition_pk = np.mean(condition[condition != 0])

    k_interp = 0.106 # Pick out a specific wavenumber for interpolation.
    interpolation = np.interp(k_interp, k['k_cen'], chosen_stat)

    print(condition_pk, interpolation)

    return