import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits
import random
import astropy
from astropy.cosmology import Planck18 as cosmo
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Serif"],
    "font.size": 16,
})

def gaussian_model(x, A, sigma):
    '''
    Gaussian model to fit.
    
    Parameters
    ----------
    x: input variable
    A: amplitude of the Gaussian curve
    sigma: standard deviation of the Gaussian
    '''
    return A * np.exp(-(x**2) / (2 * sigma**2))

def peak_fraction(d_los, A, sigma, d_threshold):
    '''
    Calculate the fraction in the area under the 
    Gaussian within a given range of LOS displacements.
    
    Parameters
    ----------
    d_los: array containing LOS displacements
    A: amplitude of the Gaussian curve
    sigma: standard deviation of the Gaussian
    d_threshold: range around the peak to consider
    '''
    p_dlos = gaussian_model(d_los, A, sigma)
    sum_peak = np.sum(p_dlos[np.abs(d_los) < d_threshold * sigma])
    return sum_peak / len(d_los)

def plot_gaussian(d_los, A, sigma):
    plt.figure()
    plt.plot(d_los, gaussian_model(d_los, A, sigma))
    
    plt.xlabel('$d_{\text{LOS}}$')
    plt.ylabel('$p(d_\text{LOS})$')
    plt.savefig('gaussian', dpi=800)

def calc_comoving_distance(redshift):
    '''Calculate the comoving distance in Mpc given a redshift.'''
    return cosmo.comoving_distance(redshift).to('Mpc').value

def downsample_catalogue(ra, dec, redshift, w_fc, fraction):
    '''
    Downsample a catalogue to use only a specified fraction.

    Parameters
    ----------
    ra: right ascension as array
    dec: declination as array
    redshift: redshift values as array
    w_fc: completeness weights as array
    fraction: float, fraction to retain (e.g., 0.1 for 10%)
    
    Return
    ------
    Downsampled catalogue
    '''
    mask = np.random.choice([False, True], len(ra), p=[1 - fraction, fraction])
    return ra[mask], dec[mask], redshift[mask], w_fc[mask]

def get_fc_pairs(redshift, w_fc):
    '''
    Determine the fibre collided galaxies.
    
    Parameters
    ----------
    redshift: galaxy redshifts as array
    w_fc: completeness weights as array
    
    Return
    ------
    fc_pairs: pairs of galaxies suffering from fibre collisions
    '''
    # Observed galaxies are FC if they have a completeness weight > 1.
    fc_condition = (w_fc > 1)
    
    fc_pairs = [(i, j) for i in range(len(redshift)) for j in range(len(redshift))
                      if fc_condition[i] and fc_condition[j] and i != j]
    
    return fc_pairs

def calc_dlos(redshift):
    '''
    Calculate the line-of-sight displacements for FC pairs.
    
    Parameter
    ---------
    redshift: galaxy redshifts as array
    
    Return
    ------
    d_los: line-of-sight displacements
    '''
    n = len(redshift)
    
    # Check if redshift array is empty (useful if data format changes).
    if n == 0:
        raise ValueError("Input redshift array is empty.")
    
    d_los = np.zeros_like(redshift, dtype=float)
    
    # Calculate LOS comoving distance.
    d_c = calc_comoving_distance(redshift)
    
    # Use only lower triangular indices to get unique pairs.
    i, j = np.tril_indices(n, k=1)
    for idx_i, idx_j in zip(i, j):
        d_los[idx_i] = d_c[idx_i] - d_c[idx_j]
        
    return d_los

def introduce_galaxy(ra, dec, redshift, w_fc, nn_index, d_peak):
    new_ra = ra[nn_index]
    new_dec = dec[nn_index]
    new_redshift = redshift[nn_index] + d_peak
    w_fc[nn_index] -= 1

    ra = np.append(ra, new_ra)
    dec = np.append(dec, new_dec)
    redshift = np.append(redshift, new_redshift)
    w_fc = np.append(w_fc, 1)
        
def reconstruction(ra_comp, dec_comp, fba_file, w_fc_threshold=1):
    # Load incomplete FITS catalogue.
    hdu = fits.open(fba_file)
    data = hdu[1].data
    
    # Extract relevant columns from the incomplete catalogue.
    ra_fba = data['RA']
    dec_fba = data['Dec']
    w_fc_fba = data['WEIGHT_COMP']
    redshift_fba = data['Z']   
        
    # Identify galaxies from the complete catalogue that are not in the incomplete catalogue.
    not_incomplete_mask = ~np.isin(ra_comp, ra_fba) & ~np.isin(dec_comp, dec_fba)

    # Concatenate RA, Dec, w_fc, and redshift arrays.
    ra_comb = np.concatenate([ra_comp[not_incomplete_mask], ra_fba])
    dec_comb = np.concatenate([dec_comp[not_incomplete_mask], dec_fba])
    w_fc_comb = np.concatenate([np.zeros(np.sum(not_incomplete_mask)), w_fc_fba])
    redshift_comb = np.concatenate([np.full(np.sum(not_incomplete_mask), np.nan), redshift_fba])
    
    ra_comb, dec_comb, w_fc_comb, redshift_comb = downsample_catalogue(
        ra_comb, dec_comb, w_fc_comb, redshift_comb, 0.03
    )
    
    ra_fba, dec_fba, w_fc_fba, redshift_fba = downsample_catalogue(
        ra_fba, dec_fba, w_fc_fba, redshift_fba, 0.03
    )
    
    # Get the FC pairs from the FBA catalogue,
    fc_pairs = get_fc_pairs(redshift_fba, w_fc_fba)
    ra, dec, redshift, w_fc = ra_fba[fc_pairs], dec_fba[fc_pairs], redshift_fba[fc_pairs], w_fc_fba[fc_pairs]
    
    # Calculate LOS displacements between FC pairs.
    print('Calculating displacements...')
    d_los = calc_dlos(redshift)
        
    sigma = np.std(d_los)
    prob = gaussian_model(d_los, A=1, sigma=sigma)
    
    np.savetxt('p_recon', np.column_stack((d_los, prob)), header='d_los p(d_los)')
    
    bins = np.arange(-3*sigma, 3*sigma, 0.1)
    plt.hist(d_los, weights=prob, bins=bins, density=True, histtype='step', color='midnightblue')
    plt.xlabel('$d_\mathrm{LOS}$ / Mpc')
    plt.ylabel('Galaxy number density / Mpc$^{-3}$')
    plt.xlim(-50, 50)
    plt.tight_layout()
    
    plt.savefig('gaussian', dpi=800)
    
    # Mitigate fibre collisions
    while np.any(w_fc_comb > 1):
        nn_index = get_fc_pairs(w_fc_comb, w_fc_threshold)
        d_peak = np.random.normal(0, popt[1])
        introduce_galaxy(ra_comb, dec_comb, redshift_comb, w_fc_comb, nn_index, d_peak)
    
    hist, bin_edges = np.histogram(d_los, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    popt, _ = curve_fit(gaussian_model, bin_centers, hist)
    
    f_peak = peak_fraction(d_los, *popt, d_threshold=40)
    plot_gaussian(d_los, *popt)
    
    save_to_fits(output_file='%s_recon' %fba_file)
    
def save_to_fits(output_file, ra, dec, redshift):
    new_table = fits.BinTableHDU.from_columns([
        fits.Column(name='RA', format='D', array=ra),
        fits.Column(name='Dec', format='D', array=dec),
        fits.Column(name='Z', format='D', array=redshift),
    ])

    new_table.writeto(output_file, overwrite=True)
      
# Load the complete catalogue FITS file.
comp_cat = 'catalogues/LRG_complete_N_clustering.dat.fits'
hdu_comp = fits.open(comp_cat)
data_comp = hdu_comp[1].data

ra_comp, dec_comp = data_comp['RA'], data_comp['Dec']

fba_file = 'catalogues/LRG_N_clustering.dat.fits'

reconstruction(ra_comp, dec_comp, fba_file, w_fc_threshold=1)
