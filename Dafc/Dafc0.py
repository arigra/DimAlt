import os
import pickle
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

def ncVaraible_to_numpy(var):
    # type(var) == netCDF4._netCDF4.Variable
    value = var[:]
    mask = (value < -99999.98) & (value > -100000.00)
    value = np.ma.MaskedArray(value, mask=mask).data
    return value


def ipixload(cdf_path, pol='vv', adc_like_I=None, adc_like_Q=None, adc_cross_I=None, adc_cross_Q=None):
    """
    python script to parse IPIX radar dataset.
    reference: http://soma.ece.mcmaster.ca/ipix/dartmouth/cdfhowto.html

    """

    #read raw data
    rootgrp = nc.Dataset(cdf_path, "r")
    rng_bins = ncVaraible_to_numpy(rootgrp.variables['range'])
    adc_data = ncVaraible_to_numpy(rootgrp.variables['adc_data'])

    adc_like_I = ncVaraible_to_numpy(rootgrp.variables['adc_like_I']) if adc_like_I is None else adc_like_I
    adc_like_Q = ncVaraible_to_numpy(rootgrp.variables['adc_like_Q']) if adc_like_Q is None else adc_like_Q
    adc_cross_I = ncVaraible_to_numpy(rootgrp.variables['adc_cross_I']) if adc_cross_I is None else adc_cross_I
    adc_cross_Q = ncVaraible_to_numpy(rootgrp.variables['adc_cross_Q']) if adc_cross_Q is None else adc_cross_Q

    # extract desired polarization
    H_txpol = 0
    V_txpol = 1
    if pol == 'hh':
        xiq = adc_data[:,H_txpol,:,[adc_like_I, adc_like_Q]]
    elif pol == 'hv':
        xiq = adc_data[:, H_txpol, :, [adc_cross_I, adc_cross_Q]]
    elif pol == 'vv':
        xiq = adc_data[:, V_txpol, :, [adc_like_I, adc_like_Q]]
    elif pol == 'vh':
        xiq = adc_data[:, V_txpol, :, [adc_cross_I, adc_cross_Q]]
    else:
        raise Exception(' ')
    xiq = np.transpose(xiq, [2, 1, 0])

    # apply corrections to I and Q data, correction is applied per range bin
    iq_mean = np.expand_dims(np.mean(xiq, 1), 1)
    iq_std = np.expand_dims(np.std(xiq, 1), 1)
    xiq = (xiq - iq_mean) / iq_std
    I = xiq[:, :, 0]
    Q = xiq[:, :, 1]
    sin_beta = np.expand_dims(np.mean(I * Q, 1), 1)
    inbal = np.arcsin(sin_beta) * 180/np.pi
    I = (I - Q * sin_beta) / np.sqrt(1 - sin_beta ** 2)

    return I, Q, iq_mean, iq_std

def parse_ipix_data(cdf_path, pol, adc_like_I, adc_like_Q, adc_cross_I, adc_cross_Q):
    # assert config.f_c == np.expand_dims(ncVaraible_to_numpy(rootgrp.variables['RF_frequency']), 0)[0]*1e9
    rootgrp = nc.Dataset(cdf_path, "r")

    f_c = np.expand_dims(ncVaraible_to_numpy(rootgrp.variables['RF_frequency']), 0)[0]*1e9
    PRI = 1 / np.expand_dims(ncVaraible_to_numpy(rootgrp.variables['PRF']), 0)[0]
    rng_bins = ncVaraible_to_numpy(rootgrp.variables['range'])
    rng_bins = np.around(rng_bins, 1)
    B = (3e8 / (2 * np.diff(rng_bins)[0]))
    # load data
    I, Q, iq_mean, iq_std = ipixload(cdf_path, pol, adc_like_I, adc_like_Q, adc_cross_I, adc_cross_Q)
    iq_mean = np.squeeze(iq_mean, 1)
    iq_std = np.squeeze(iq_std, 1)
    X = np.expand_dims(iq_std[:,0], 1)*I + 1j*np.expand_dims(iq_std[:,1], 1)*Q

    return X, PRI, rng_bins, B


