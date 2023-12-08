import nbodykit.lab
import os
import numpy as np
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import astropy.cosmology as astro
from astropy.coordinates import SkyCoord
from tqdm import tqdm
import sys

cosmo = nbodykit.lab.cosmology.Cosmology(
    h=0.6736, Omega0_b=0.04930, Omega0_cdm=0.2645,
    A_s=2.083e-09, n_s=0.9649
)

def read_cat(cap, comp, cat, tracer, real):
    '''
    Read in the catalogues and cut them for the appropriate redshifts.
    
    Parameters
    ----------
    cap: 'N' or 'S'
    comp: 'complete' or ''
    cat: 'dat' or 'ran'
    tracer: 'LRG', 'ELG' or 'QSO'
    real: realisation number
    
    Return 
    ------
    cat_data: cut catalogue
    '''

    base_dir = 'Y1/mock%s/LSScats' %real

    if comp == 'complete':
        if cat == 'dat':
            filename = '%s_%s_%s_clustering.%s.fits' % (tracer, comp, cap, cat)
        else:
            filename = '%s_%s_%s_0_clustering.%s.fits' % (tracer, comp, cap, cat)
    else:
        if cat == 'dat':
            filename = '%s_%s_clustering.%s.fits' % (tracer, cap, cat)
        else:
            filename = '%s_%s_0_clustering.%s.fits' % (tracer, cap, cat)
    
    cat_data = nbodykit.lab.FITSCatalog(os.path.join(base_dir, filename))
    
    if tracer == 'LRG':
        cat_data = cat_data[(cat_data['Z'] > 0.4) & (cat_data['Z'] < 1.1)]
    elif tracer == 'ELG':
        cat_data = cat_data[(cat_data['Z'] > 0.8) & (cat_data['Z'] < 1.6)]
    elif tracer == 'QSO':
        cat_data = cat_data[(cat_data['Z'] > 0.8) & (cat_data['Z'] < 2.1)]
    
    return cat_data

def save_corrected(true_cap, cat, tracer, real, data):
    '''
    Save nearest neighbour corrected catalogues.
    
    Parameters
    ----------
    true_cap: 'N' or 'S' (the corrected cap)
    cat: 'dat' or 'ran'
    tracer: 'LRG', 'ELG' or 'QSO'
    real: realisation number
    data: transformed catalogue to save
    '''
    with open('./storage/input/catalogues/%s/%s_%s_%s_%s_corrected' % 
              (tracer, tracer, true_cap, cat, real), 'w') as f:
        np.savetxt(f, np.column_stack(data), header='x y z nz ws wc')
        f.close()

    return

def save_uncorrected(comp, true_cap, cat, tracer, real, data):
    '''
    Save catalogues without a correction (complete/incomplete).
    
    Parameters
    ----------
    comp: 'complete' or 'incomplete'
    true_cap: 'N' or 'S' (the corrected cap)
    cat: 'dat' or 'ran'
    tracer: 'LRG', 'ELG' or 'QSO'
    real: realisation number
    data: transformed catalogue to save
    '''
    if comp == 'complete':
        with open('./storage/input/catalogues/%s/%s_%s_%s_%s_complete' % 
                  (tracer, tracer, true_cap, cat, real), 'w') as f:
            np.savetxt(f, np.column_stack(data), header='x y z nz wc')
            f.close()
    if comp == '':
        with open('./storage/input/catalogues/%s/%s_%s_%s_%s_incomplete' % 
                  (tracer, tracer, true_cap, cat, real), 'w') as f:
            np.savetxt(f, np.column_stack(data), header='x y z nz wc')
            f.close()

    return

def sky_to_cartesian(cat_data, corr):
    '''
    Determine galaxy positions in Cartesian coordinates from RA, Dec.
    Express data in array with Triumvirate-compatible columns.

    Parameters
    ----------
    cat_data: catalogue for conversion
    corr: whether or not the catalogue is corrected for systematics
    
    Return
    ------
    data: catalogue as array with appropriate columns
    '''
    
    # Pick out RA, Dec, redshift.
    ra = cat_data['RA']
    dec = cat_data['DEC']
    redshift = cat_data['Z']

    # Broadcast RA, Dec, redshift as dask arrays. 
    ra, dec, redshift = da.broadcast_arrays(ra, dec, redshift)
    ra = da.deg2rad(ra)
    dec = da.deg2rad(dec)

    # Convert RA, Dec to Cartesian coordinates on the unit sphere.
    x = da.cos(dec) * da.cos(ra)
    y = da.cos(dec) * da.sin(ra)
    z = da.sin(dec)

    # Pick out ws and nz as numpy arrays.
    wc = np.array(cat_data['WEIGHT_FKP'], dtype=np.float32)
    nz = np.array(cat_data['NZ'], dtype=np.float32)

    # Use comoving distance to find true Cartesian coordinates for each galaxy. 
    r = cosmo.comoving_distance(redshift)
    x, y, z = r * x, r * y, r * z
    x, y, z = np.array(x, dtype=np.float32),\
              np.array(y, dtype=np.float32),\
              np.array(z, dtype=np.float32)

    if corr == '':
        data = [x, y, z, nz, wc]
    else:
        ws = np.array(cat_data['WEIGHT_COMP'], dtype=np.float32)
        data = [x, y, z, nz, ws, wc]

    return data 

def conversion(comp, cat, corr, tracer, real):
    '''
    Create Cartesian position columns and save new files to specified directory.

    Parameters
    ----------
    comp: 'complete' or ''
    cat: 'dat' or 'ran'
    corr: 'corrected' or ''
    tracer: 'LRG', 'ELG' or 'QSO'
    real: realisation number
    '''
    cat_n = read_cat('N', comp, cat, tracer, real)
    cat_s = read_cat('S', comp, cat, tracer, real)

    # Combine the N and S catalogues.
    cat_data = nbodykit.lab.transform.ConcatenateSources(cat_n, cat_s)

    # Pick out the right ascension.
    ra = np.array(cat_data['RA']).T
    indx = (ra<300)*(ra>100)

    data_n = sky_to_cartesian(cat_data[indx], corr)
    save_uncorrected(comp, 'N', cat, tracer, real, data_n)

    data_s = sky_to_cartesian(cat_data[~indx], corr)
    save_uncorrected(comp, 'S', cat, tracer, real, data_s)

    return

tracers = ['LRG', 'ELG', 'QSO']
caps = ['N', 'S']
cats = ['dat', 'ran']

conversion('complete', 'dat', '', 'LRG', '1')
