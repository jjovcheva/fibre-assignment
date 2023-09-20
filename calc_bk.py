from triumvirate.logger import setup_logger
from triumvirate.parameters import ParameterSet
from triumvirate.catalogue import ParticleCatalogue
from triumvirate.threept import (
    compute_3pcf, compute_3pcf_in_gpp_box, compute_3pcf_window,
    compute_bispec, compute_bispec_in_gpp_box,
)

import numpy as np
import dask.array as da
import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm
import ruamel.yaml

logger = setup_logger()
yaml = ruamel.yaml.YAML()

# Specify the different tracers, galactic caps, and multipoles.
tracers = ['LRG', 'ELG', 'QSO']
comps = ['complete', 'incomplete', 'corrected']
ell = ['000', '202', '220', '112', '404']
caps = ['N', 'S']

# i = int(sys.argv[1])
i = 0
tracer, comp = tracers[0], comps[0]
filename = 'storage/input/params/pk_%s.yml' %comp

for cap in caps:
    with open('storage/input/params/pk_%s.yml' %comp) as fp:
        data = yaml.load(fp)
        
        data['directories']['catalogues'] =\
            'storage/input/catalogues/%s' %tracer

        data['directories']['measurements'] =\
            'storage/output/measurements/pk%s/%s' %(ell[0], tracer)

        data['files']['data_catalogue'] =\
            '%s_%s_dat_%s_%s' %(tracer, cap, i, comp)
        data['files']['rand_catalogue'] =\
            '%s_%s_ran_%s_%s' %(tracer, cap, i, comp)

        data['tags']['output'] = '%s_%s_%s_%s' %(tracer, cap, i, comp)

        data['degrees']['ELL'] = ell[0]
            
        with open('storage/input/params/pk_%s.yml' %comp, 'w+') as fp:
            yaml.dump(data, fp)

    pars = ParameterSet(filename, logger=logger)

    if comp == 'corrected':
        names = ['x', 'y', 'z', 'nz', 'ws', 'wc']
    else: 
        names = ['x', 'y', 'z', 'nz', 'wc']

    cat_data = ParticleCatalogue.read_from_file(
        "{}/{}".format(
            pars['directories']['catalogues'],
            pars['files']['data_catalogue']
        ),
        names=names, logger=logger)

    cat_rand = ParticleCatalogue.read_from_file(
        "{}/{}".format(
            pars['directories']['catalogues'],
            pars['files']['rand_catalogue']
        ),
        names=names, logger=logger)

    measurements = compute_bispec(
        cat_data, cat_rand, paramset=pars, save='.txt', logger=logger)