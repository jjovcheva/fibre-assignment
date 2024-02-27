import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits
import random
from astropy.cosmology import Planck18 as cosmo
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Serif"],
    "font.size": 16,
})

def gaussian_model(x, A, sigma):
    '''
    Gaussian model function.
    
    Parameters
    ----------
    x : array_like
        Input variable.
    A : float
        Amplitude of the Gaussian curve.
    sigma : float
        Standard deviation of the Gaussian curve.
    
    Returns
    -------
    array_like
        Values of the Gaussian function evaluated at input points.
    '''
    return A * np.exp(-(x**2) / (2 * sigma**2))

def peak_fraction(d_los, A, sigma, d_threshold):
    '''
    Calculate the fraction of the area under the Gaussian curve within a given range of LOS displacements.
    
    Parameters
    ----------
    d_los : array_like
        Array containing LOS displacements.
    A : float
        Amplitude of the Gaussian curve.
    sigma : float
        Standard deviation of the Gaussian curve.
    d_threshold : float
        Range around the peak to consider.
    
    Returns
    -------
    float
        Fraction of the area under the Gaussian curve within the specified range.
    '''
    p_dlos = gaussian_model(d_los, A, sigma)
    sum_peak = np.sum(p_dlos[np.abs(d_los) < d_threshold * sigma])
    return sum_peak / len(d_los)

def plot_gaussian(d_los, A, sigma):
    '''
    Plot the Gaussian curve.
    
    Parameters
    ----------
    d_los : array_like
        Array containing LOS displacements.
    A : float
        Amplitude of the Gaussian curve.
    sigma : float
        Standard deviation of the Gaussian curve.
    '''
    plt.figure()
    plt.plot(d_los, gaussian_model(d_los, A, sigma))
    plt.xlabel('$d_{\mathrm{LOS}}$')
    plt.ylabel('$p(d_{\mathrm{LOS}})$')
    plt.savefig('gaussian', dpi=800)

def calc_comoving_distance(redshift):
    '''
    Calculate the comoving distance in Mpc given a redshift.
    
    Parameters
    ----------
    redshift : array_like
        Redshift values.
    
    Returns
    -------
    array_like
        Comoving distance corresponding to the input redshifts.
    '''
    return cosmo.comoving_distance(redshift).to('Mpc').value

def downsample_catalogue(ra, dec, redshift, w_fc, fraction):
    '''
    Downsample a catalogue to use only a specified fraction.
    
    Parameters
    ----------
    ra : array_like
        Right ascension.
    dec : array_like
        Declination.
    redshift : array_like
        Redshift values.
    w_fc : array_like
        Completeness weights.
    fraction : float
        Fraction to retain (e.g., 0.1 for 10%).
    
    Returns
    -------
    tuple
        Downsampled catalogue containing RA, Dec, redshift, and weights.
    '''
    mask = np.random.choice([False, True], len(ra), p=[1 - fraction, fraction])
    return ra[mask], dec[mask], redshift[mask], w_fc[mask]

def get_resolved_pairs(redshift, w_fc):
    '''
    Determine the resolved galaxy pairs.
    
    Parameters
    ----------
    redshift: array-like
        Galaxy redshifts.
    w_fc: array-like
        Completeness weights.
    
    Return
    ------
    resolved_pairs: list
        List of resolved galaxy pairs.
    '''
    n = len(redshift)
    resolved_pairs = [(i, j) for i in range(n) for j in range(i+1, n) if w_fc[i] <= 1 and w_fc[j] <= 1]
    return resolved_pairs

def calc_dlos(ra, dec, redshift):
    '''
    Calculate the line-of-sight displacements for resolved galaxy pairs.
    
    Parameter
    ---------
    ra: array-like
        Array containing right ascension values.
    dec: array-like
        Array containing declination values.
    redshift: array-like
        Array containing redshift values.
    
    Return
    ------
    d_los: array-like
        Array containing line-of-sight displacements.
    '''
    n = len(ra)
    d_los = np.zeros_like(redshift, dtype=float)
    
    # Calculate LOS comoving distance.
    d_c = calc_comoving_distance(redshift)
    
    # Use only lower triangular indices to get unique pairs.
    i, j = np.tril_indices(n, k=1)
    for idx_i, idx_j in zip(i, j):
        d_los[idx_i] = d_c[idx_i] - d_c[idx_j]
        
    return d_los

def introduce_galaxy(ra, dec, redshift, w_fc, nn_index, d_peak):
    '''
    Introduce a galaxy to mitigate fibre collisions.
    
    Parameters
    ----------
    ra : array_like
        Right ascension.
    dec : array_like
        Declination.
    redshift : array_like
        Redshift values.
    w_fc : array_like
        Completeness weights.
    nn_index : array_like
        Nearest neighbor indices.
    d_peak : float
        Displacement peak.
    '''
    new_ra = ra[nn_index]
    new_dec = dec[nn_index]
    new_redshift = redshift[nn_index] + d_peak
    w_fc[nn_index] -= 1

    ra = np.append(ra, new_ra)
    dec = np.append(dec, new_dec)
    redshift = np.append(redshift, new_redshift)
    w_fc = np.append(w_fc, 1)

def reconstruction(ra_comp, dec_comp, fba_file, w_fc_threshold=1):
    '''
    Perform reconstruction of galaxies to mitigate fibre collisions.
    
    Parameters
    ----------
    ra_comp : array_like
        Right ascension from the complete catalogue.
    dec_comp : array_like
        Declination from the complete catalogue.
    fba_file : str
        Filepath of the incomplete catalogue.
    w_fc_threshold : float, optional
        Weight threshold for fibre collisions (default is 1).
    '''
    # Load the incomplete FITS catalogue.
    hdu = fits.open(fba_file)
    data = hdu[1].data

    ra_fba = data['RA']
    dec_fba = data['Dec']
    w_fc_fba = data['WEIGHT_COMP']
    redshift_fba = data['Z']

    # Identify galaxies from the complete catalogue that are not in the incomplete catalogue.
    not_incomplete_mask = ~np.isin(ra_comp, ra_fba) & ~np.isin(dec_comp, dec_fba)

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

    res_pairs = get_resolved_pairs(redshift_fba, w_fc_fba)
    ra, dec, redshift, w_fc = ra_fba[res_pairs], dec_fba[res_pairs], redshift_fba[res_pairs], w_fc_fba[res_pairs]

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

    while np.any(w_fc_comb > 1):
        nn_index = get_resolved_pairs(w_fc_comb, w_fc_threshold)
        d_peak = np.random.normal(0, popt[1])
        introduce_galaxy(ra_comb, dec_comb, redshift_comb, w_fc_comb, nn_index, d_peak)

    hist, bin_edges = np.histogram(d_los, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    popt, _ = curve_fit(gaussian_model, bin_centers, hist)
    f_peak = peak_fraction(d_los, *popt, d_threshold=40)
    plot_gaussian(d_los, *popt)

    save_to_fits(output_file='%s_recon' %fba_file)

def save_to_fits(output_file, ra, dec, redshift):
    '''
    Save galaxy data to a FITS file.
    
    Parameters
    ----------
    output_file : str
        Output filename.
    ra : array_like
        Right ascension.
    dec : array_like
        Declination.
    redshift : array_like
        Redshift values.
    '''
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