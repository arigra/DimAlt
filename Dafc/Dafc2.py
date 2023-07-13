import argparse
import sys
from collections import OrderedDict
import datetime
import scipy
import os
import shutil
import glob
import pickle
from pathlib import Path
import tensorflow as tf
import logging
import mlflow
import numpy as np
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
import commentjson
from random import randint
from bunch import Bunch
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, Callback, EarlyStopping
logger = logging.getLogger("logger")
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Input
from tensorflow.keras.layers import Dropout,  Activation, LeakyReLU, AveragePooling1D



def get_dataset_ipix(config, apply_tf_preprocess_pipe=True, split_data=True, return_dict=False):

    if config.ipix_cv_mode:
        # load train and validation
        data_dict_per_file = {}
        cdf_files_list = [f for f in os.listdir(config.ipix_pkl_path_dir) if not f.startswith('.')]
        cdf_files_list = [x for x in cdf_files_list if x not in config.ipix_pkl_cv_hold_out]
        for cdf_file in cdf_files_list:
            config.ipix_pkl_path = os.path.join(config.ipix_pkl_path_dir, cdf_file)
            # read raw data and convert to fast-time x slow-time complex data
            c_tensor_total, rng_bins_ipix, clutter_vel = read_data_ipix(config)
            config.r_0_max = rng_bins_ipix[-1]
            M_valid = int(config.M_valid / len(cdf_files_list))
            M0_valid = int(config.without_target_ratio_test * M_valid)
            M1_valid = M_valid
            M_train = int(config.M_train / len(cdf_files_list))
            M0_train = int(config.without_target_ratio * M_train)
            M1_train = M_train
            # generate tf.data.Dataset objects
            data = {}
            c_tensor_total_valid = c_tensor_total[:, int(0.9 * c_tensor_total.shape[1]):]
            data['valid'] = gen_ipix_pipeline_dataset(c_tensor_total_valid, clutter_vel, config, M0_valid, M1_valid)

            c_tensor_total_train = c_tensor_total[:, :int(0.9 * c_tensor_total.shape[1])]
            data['train'] = gen_ipix_pipeline_dataset(c_tensor_total_train, clutter_vel, config, M0_train, M1_train)

            if apply_tf_preprocess_pipe:
                # add data pipeline functions (maps)
                for set_type in ['train', 'valid']:
                    data[set_type] = tf_dataset_pipeline(config, data[set_type])

            data_dict_per_file[cdf_file] = data

        data = {}
        for set_type in ['train', 'valid']:
            data[set_type] = data_dict_per_file[cdf_files_list[0]][set_type]
            for cdf_file in cdf_files_list[1:]:
                data[set_type] = data[set_type].concatenate(data_dict_per_file[cdf_file][set_type])

        # load test
        data_dict_per_file = {}
        for cdf_file in config.ipix_pkl_cv_hold_out:
            assert cdf_file not in cdf_files_list
            config.ipix_pkl_path = os.path.join(config.ipix_pkl_path_dir, cdf_file)
            c_tensor_total_test, rng_bins_ipix, clutter_vel = read_data_ipix(config)
            M_test = int(config.M_test / len(config.ipix_pkl_cv_hold_out))
            M0_test = int(config.without_target_ratio_test * M_test)
            M1_test = M_test
            data['test'] = gen_ipix_pipeline_dataset(c_tensor_total_test, clutter_vel, config, M0_test, M1_test)

            if apply_tf_preprocess_pipe:
                data['test'] = tf_dataset_pipeline(config, data['test'])

            data_dict_per_file[cdf_file] = data['test']

        data['test'] = data_dict_per_file[config.ipix_pkl_cv_hold_out[0]]
        for cdf_file in config.ipix_pkl_cv_hold_out[1:]:
            data['test'] = data['test'].concatenate(data_dict_per_file[cdf_file])

    else:

        data_dict_per_file = {}
        cdf_files_list = [f for f in os.listdir(config.ipix_pkl_path_dir) if not f.startswith('.')]
        assert len(cdf_files_list) > 0
        # cdf_files_list = ['19980227_221025_ANTSTEP_pol_hh.pkl']
        for cdf_file in cdf_files_list:
            config.ipix_pkl_path = os.path.join(config.ipix_pkl_path_dir, cdf_file)
            # read raw data and convert to fast-time x slow-time complex data
            c_tensor_total, rng_bins_ipix, clutter_vel = read_data_ipix(config)
            config.r_0_max = rng_bins_ipix[-1]
            M_test = int(config.M_test / len(cdf_files_list))
            M0_test = int(config.without_target_ratio_test * M_test)
            M1_test = M_test
            M_valid = int(config.M_valid / len(cdf_files_list))
            M0_valid = int(config.without_target_ratio_test * M_valid)
            M1_valid = M_valid
            M_train = int(config.M_train / len(cdf_files_list))
            M0_train = int(config.without_target_ratio * M_train)
            M1_train = M_train

            # generate tf.data.Dataset objects
            data = {}
            if split_data:
                # test = [0.9, 1.0], valid = [0.85, 0.9], train = [0.0, 0.85]
                c_tensor_total_test = c_tensor_total[:,int(0.9 * c_tensor_total.shape[1]):]
                data['test'] = gen_ipix_pipeline_dataset(c_tensor_total_test, clutter_vel, config, M0_test, M1_test)

                c_tensor_total_valid = c_tensor_total[:, int(0.85 * c_tensor_total.shape[1]): int(0.9 * c_tensor_total.shape[1])]
                data['valid'] = gen_ipix_pipeline_dataset(c_tensor_total_valid, clutter_vel, config, M0_valid, M1_valid)

                c_tensor_total_train = c_tensor_total[:, :int(0.85 * c_tensor_total.shape[1])]
                data['train'] = gen_ipix_pipeline_dataset(c_tensor_total_train, clutter_vel, config, M0_train, M1_train)
            else:
                M1_test = config.M_test
                M0_test = int(config.M_test * config.without_target_ratio_test)
                data['test'] = gen_ipix_pipeline_dataset(c_tensor_total, clutter_vel, config, M0_test, M1_test)


            if apply_tf_preprocess_pipe:
                # add data pipeline functions (maps)
                for set_type in ['train', 'valid', 'test']:
                    data[set_type] = tf_dataset_pipeline(config, data[set_type])

            data_dict_per_file[cdf_file] = data

        if return_dict:
            return data_dict_per_file
        else:
            data = {}
            for set_type in ['train', 'valid', 'test']:
                data[set_type] = data_dict_per_file[cdf_files_list[0]][set_type]
                for cdf_file in cdf_files_list[1:]:
                    data[set_type] = data[set_type].concatenate(data_dict_per_file[cdf_file][set_type])

    return data

#------------------------------------------#

def get_dataset_compund_gaussian(config, apply_tf_preprocess_pipe=True):
    data = {}
    M1_train = config.M_train
    M0_train = int(config.M_train * config.without_target_ratio)
    M1_valid = config.M_valid
    M0_valid = int(config.M_valid * config.without_target_ratio_test)
    M1_test = config.M_test
    M0_test = int(config.M_test * config.without_target_ratio_test)

    if config.embedded_target:
        assert config.compound_gaussian_single_clutter_vel
    data['train'] = gen_compound_gaussian_pipeline_dataset(config, M0_train, M1_train)
    data['valid'] = gen_compound_gaussian_pipeline_dataset(config, M0_valid, M1_valid)
    data['test'] = gen_compound_gaussian_pipeline_dataset(config, M0_test, M1_test)

    if apply_tf_preprocess_pipe:
        # add data pipeline functions (maps)
        for set_type in ['train', 'valid', 'test']:
            data[set_type] = tf_dataset_pipeline(config, data[set_type])

    return data

#------------------------------------------#

def get_dataset_wgn(config, apply_tf_preprocess_pipe=True):

    data = {}
    M1_train = config.M_train
    M0_train = int(config.M_train * config.without_target_ratio)
    M1_valid = config.M_valid
    M0_valid = int(config.M_valid * config.without_target_ratio_test)
    M1_test = config.M_test
    M0_test = int(config.M_test * config.without_target_ratio_test)

    data['train'] = gen_wgn_pipeline_dataset(config, M0_train, M1_train)
    data['valid'] = gen_wgn_pipeline_dataset(config, M0_valid, M1_valid)
    data['test'] = gen_wgn_pipeline_dataset(config, M0_test, M1_test)

    if apply_tf_preprocess_pipe:
        # add data pipeline functions (maps)
        for set_type in ['train', 'valid', 'test']:
            data[set_type] = tf_dataset_pipeline(config, data[set_type])

    return data

#------------------------------------------#

def get_model_output_dim(data, set_type):
    if len(list(data['train'].element_spec[1].shape)) > 1:
        return [list(data['train'].element_spec[1].shape)[0]]
    if type(data[set_type].element_spec[1]) == type(tuple()):
        return list(data[set_type].element_spec[1][0].shape)
    else:
        return [list(data[set_type].element_spec[1].shape)[0]]

def get_model_input_dim(data, set_type):
    if isinstance(data[set_type].element_spec[0], tuple):
        model_input_dim = []
        for spec in data[set_type].element_spec[0]:
            model_input_dim.append(list(spec.shape))
    else:
        model_input_dim = list(data[set_type].element_spec[0].shape)

    return model_input_dim

#------------------------------------------#

def make_iterators(data, config):

    M_train = len(data['train'])
    print('make_iterators(): M_train: {}'.format(M_train))
    data['train'] = data['train'].shuffle(M_train, reshuffle_each_iteration=True)


    train_iter = data['train'].batch(config.batch_size, drop_remainder=True).prefetch(config.batch_size)
    valid_iter = data['valid'].batch(config.batch_size)
    test_iter = data['test'].batch(config.batch_size)

    iterators = {'train': train_iter, 'valid': valid_iter, 'test': test_iter}
    return iterators

#------------------------------------------#

def load_data(config, use_make_iterators=True, apply_tf_preprocess_pipe=True):
    model_input_dim_set = 'train'
    if config.data_name == "ipix":
        data = get_dataset_ipix(config, apply_tf_preprocess_pipe=apply_tf_preprocess_pipe)
        model_input_dim_set = 'test'
    elif config.data_name == "compound_gaussian":
        data = get_dataset_compund_gaussian(config, apply_tf_preprocess_pipe=apply_tf_preprocess_pipe)
        if config.compound_gaussian_add_wgn:
            data_wgn = get_dataset_wgn(config, apply_tf_preprocess_pipe=apply_tf_preprocess_pipe)
            for key in data.keys():
                # rd_signal , label_tensor, (param_val_tensor, scnr_tensor, gamma_shape, clutter_vel, clutter_label_tensor)
                data[key] = data[key].map(lambda x0, x1, x2: (x0, x1, (x2[0], x2[1], x2[2], tf.constant(0.0), tf.constant(0.0))))
                data[key] = data[key].concatenate(data_wgn[key])
    elif config.data_name == "wgn":
        data = get_dataset_wgn(config, apply_tf_preprocess_pipe=apply_tf_preprocess_pipe)
    else:
        raise Exception(' ')

    # set model_input_dim
    config.model_input_dim = get_model_input_dim(data, model_input_dim_set)
    config.model_output_dim = get_model_output_dim(data, model_input_dim_set)

    if use_make_iterators:
        # make data iterators (shuffle,batch,etc.)
        data_iterators = make_iterators(data, config)
        return config, data_iterators
    else:
        return config, data
    
#------------------------------------------#

def read_data_ipix(config):
    with open(config.ipix_pkl_path, 'rb') as handle:
        ipix_data =pickle.load(handle)
        PRI = ipix_data['PRI']
        B = ipix_data['B']
        rng_bins = ipix_data['rng_bins']
        adc_data = ipix_data['adc_data']

    rng_bins = rng_bins[:config.ipix_max_nrange_bins]
    adc_data = adc_data[:config.ipix_max_nrange_bins, :]
    if not config.ipix_file_range_bins:
        assert config.N <= len(rng_bins) * 2
        adc_data = adc_data[:config.N // 2, :]
        rng_bins = rng_bins[:config.N // 2]
    else:
        config.N = len(rng_bins) * 2
    assert config.N == len(rng_bins) * 2
    config.T_PRI = PRI
    config.B_chirp = B
    if '19980205_191043' in config.ipix_pkl_path:
        # cut weird zero part of this file
        adc_data = adc_data[:, :50000]
    # convert to fast-time x slow-time data
    rng_bins = rng_bins - rng_bins[0]
    config.r_0_max = rng_bins[-1]
    clutter_omega_r = ((2 * np.pi) / config.N) * ((2*B) / 3e8) * rng_bins
    # workaround to prevent GPU overflow in multiple iterations
    try:
        clutter_range_steering_tensor = tf.math.exp(-1j * tf.cast(tf.expand_dims(tf.range(config.N, dtype=tf.float32), -1) * tf.expand_dims(clutter_omega_r, 0), dtype=tf.complex128))
        c_tensor = clutter_range_steering_tensor @ adc_data
    except:
        clutter_range_steering_tensor = np.exp(-1j * tf.cast(tf.expand_dims(tf.range(config.N, dtype=tf.float32), -1) * tf.expand_dims(clutter_omega_r, 0), dtype=tf.complex128))
        c_tensor = clutter_range_steering_tensor @ adc_data

    """
    estimate clutter Doppler frequency using welch method: 
        c_tensor[i] = e ^ {j 2 \pi f_d kT_PRI}
        f_d = (2 * f_c * clutter_vel) / c
    """

    Pxx_den_list = []
    for i in range(adc_data.shape[0]):
        f, Pxx_den = scipy.signal.welch(adc_data[i], 1 / PRI, return_onesided=False)
        Pxx_den_list.append(Pxx_den)

    PSD = np.mean(np.array(Pxx_den_list), 0)
    PSD = PSD / np.max(PSD)
    clutter_fd = f[np.argmax(PSD)]
    clutter_vel = -(3e8 * clutter_fd) / (2 * 9.39e9)

    return c_tensor, rng_bins, clutter_vel

#------------------------------------------#

def gen_ipix_pipeline_dataset(c_tensor_total, clutter_vel, config, M0, M1):

    def gen_ipix_frame2d(ind):
        c_tensor = tf.image.random_crop(c_tensor_total, [config.N, config.K])
        clutter_vel_local = clutter_vel
        if config.ipix_random_shift_doppler:
            shift_min = tf.cast(clutter_vel_local - config.v_r_min, dtype=tf.float32)
            shift_max = tf.cast(config.v_r_max - clutter_vel_local, dtype=tf.float32)
            doppler_shift_v = tf.cast(tf.random.uniform([], -shift_min, shift_max), dtype=tf.complex128)
            clutter_vel_local = clutter_vel + tf.cast(tf.math.real(doppler_shift_v), dtype=tf.float32)
            shift_factor = tf.expand_dims(tf.math.exp(-1j * 2 * np.pi * ((2 * config.f_c * doppler_shift_v) / 3e8) * config.T_PRI * tf.cast(tf.range(c_tensor.shape[1]), dtype=tf.complex128) ), 0 )
            c_tensor = c_tensor * shift_factor

            # fig, ax = plt.subplots(figsize=(6, 6))
            # im = ax.imshow(np.log10(tf.signal.fftshift(tf.abs(tf.signal.ifft2d(c_tensor)), axes=(0, 1)))[32:, :], interpolation='none', extent=[np.min(recon_vec_vel), np.max(recon_vec_vel), np.max(recon_vec_rng), np.min(recon_vec_rng)], aspect="auto")
            # plt.xlabel("Doppler Velocity [m/s]")
            # plt.ylabel("Normalized Range [m]")
            # plt.show()
        if with_target is False:
            param_val_tensor = tf.ones(config.num_targets) * -1000.0, tf.ones(config.num_targets) * -1000.0
            return c_tensor, tf.zeros((recon_vec_rng.shape[0], recon_vec_vel.shape[0]), dtype=tf.int64), \
                   param_val_tensor, tf.ones(config.num_targets) * -1000.0, tf.constant(0.0), clutter_vel_local, tf.constant(0.0)
        else:
            cn_norm = tf.abs(tf.linalg.norm(c_tensor))
            rd_signal, label_tensor, param_val_tensor, scnr_tensor = gen_target_matrix(config, cn_norm, clutter_vel_local, N, K, recon_vec_rng, recon_vec_vel)

            return rd_signal + c_tensor, label_tensor, param_val_tensor, scnr_tensor, tf.constant(0.0), clutter_vel_local, tf.constant(0.0)

    def get_ipix_tfds(M_tfds):
        _res = gen_ipix_frame2d(0)
        tfds = tf.data.Dataset.range(0, M_tfds).map(gen_ipix_frame2d, num_parallel_calls=-1)

        return tfds

    assert config.N == c_tensor_total.shape[0]
    assert not (config.SCNR_db_random_constant and config.SCNR_db_random_choice)
    N = config.N
    K = config.K

    recon_vec_rng = tf.cast(get_reconstruction_point_cloud_vec(config, param_ind=0), dtype=tf.float32)
    recon_vec_vel = tf.cast(get_reconstruction_point_cloud_vec(config, param_ind=1), dtype=tf.float32)

    with_target = False
    tfds0 = get_ipix_tfds(M0)
    with_target = True
    tfds1 = get_ipix_tfds(M1)

    assert tfds1.element_spec[0] == tfds0.element_spec[0] and tfds1.element_spec[1] == tfds0.element_spec[1]
    tfds = tfds1.concatenate(tfds0)
    tfds = tfds.map(split_auxillary_structure)


    return tfds

#------------------------------------------#

def get_reconstruction_point_cloud_vec(config, param_ind):

    if config.point_cloud_reconstruction_fft_dims:
        N = config.point_cloud_reconstruction_fft_dim_factor * config.N
        config.B_chirp = config.point_cloud_reconstruction_fft_dim_factor * config.B_chirp # multiply to rescale range dimension
        K = config.point_cloud_reconstruction_fft_dim_factor * config.K
        L = config.point_cloud_reconstruction_fft_dim_factor * config.L

        range_res, vel_res, recon_vec_rng, recon_vec, azimuth_bins_values = get_fft_resolutions(config, [1, N, K, L], T_PRI=config.T_PRI)
        bin_values_list, valid_bins_list = get_valid_2d_bins(config, [N, K], recon_vec_rng, recon_vec)
        range_bins_values = bin_values_list[0]
        vel_bins_values = bin_values_list[1]

        config.B_chirp = config.B_chirp /  config.point_cloud_reconstruction_fft_dim_factor
        if param_ind == 0:
            return range_bins_values
        elif param_ind == 1:
            return vel_bins_values
        else:
            raise Exception('  ')
    else:
        raise Exception('get_reconstruction_point_cloud_res(): Unsupported...')


#------------------------------------------#

def get_fft_resolutions(config, data_shape, T_PRI=None):
    assert len(data_shape) == 4
    T_PRI = data_shape[1] * (1 / config.f_s) + config.T_idle if T_PRI is None else T_PRI
    vel_res = 3e8 / (2 * config.f_c * data_shape[2] * T_PRI)
    range_res = 3e8 / (2 * config.B_chirp)
    range_bins_values = np.array([range_res * (i - data_shape[1] // 2) for i in range(data_shape[1])])
    vel_bins_values = np.array([vel_res * (i - data_shape[2] // 2) for i in range(data_shape[2])])

    azimuth_bins_values = np.arcsin([(2.0 * (i - data_shape[3] // 2)) / data_shape[3]
                                          for i in range(data_shape[3])]) * (180.0 / np.pi)

    return range_res, vel_res, range_bins_values, vel_bins_values, azimuth_bins_values

#------------------------------------------#


def get_valid_2d_bins(config, full_shape, range_bins_values, vel_bins_values):
    assert len(full_shape) == 2
    valid_vel_bins = \
    np.where(np.logical_and(vel_bins_values >= config.v_0_min, vel_bins_values <= config.v_0_max))[0]
    if valid_vel_bins[-1] < full_shape[1] - 1:
        valid_vel_bins = np.append(valid_vel_bins, valid_vel_bins[-1] + 1)
    if valid_vel_bins[0] > 0:
        valid_vel_bins = np.insert(valid_vel_bins, 0, valid_vel_bins[0] - 1)
    vel_bins_values = vel_bins_values[valid_vel_bins]

    valid_range_bins = np.where(np.logical_and(range_bins_values >= config.r_0_min, range_bins_values <= config.r_0_max))[0]
    # if valid_range_bins[-1] < full_shape[0] - 1:
    #     valid_range_bins = np.append(valid_range_bins, valid_range_bins[-1] + 1)
    # if valid_range_bins[0] > 0:
    #     valid_range_bins = np.insert(valid_range_bins, 0, valid_range_bins[0] - 1)
    range_bins_values = range_bins_values[valid_range_bins]

    return [range_bins_values, vel_bins_values], [valid_range_bins, valid_vel_bins]

#------------------------------------------#

def gen_target_matrix(config, cn_norm, clutter_vel_local, N, K, recon_vec_rng, recon_vec_vel):

    # randomly determine the number of targets or set it to a constant value
    if config.random_num_targets:
        targets_num = tf.random.uniform([1, ], minval=1, maxval=config.num_targets + 1, dtype=tf.int64)
    else:
        targets_num = tf.cast(tf.constant([config.num_targets]), dtype=tf.int64)
    
    # Generate target velocities within specified range considering the presence of embedded targets
    if config.embedded_target:
        targets_vel = tf.random.uniform(targets_num, tf.maximum(clutter_vel_local - config.embedded_target_vel_offset, config.v_0_min),
                                                     tf.minimum(clutter_vel_local + config.embedded_target_vel_offset, config.v_0_max))
    else:
        targets_vel = tf.random.uniform(targets_num, config.v_0_min, config.v_0_max)
    
    # Generate target ranges within specified range
    targets_rng = tf.random.uniform(targets_num, config.r_0_min, config.r_0_max)
    
    # compute doppler and range target frequencies
    targets_omega_d = tf.cast(2 * np.pi * config.T_PRI * ((2 * config.f_c * targets_vel) / 3e8), dtype=tf.complex128)
    targets_omega_r = tf.cast(2 * np.pi * ((2 * config.B_chirp * targets_rng) / (3e8 * N)), dtype=tf.complex128)
    
    # compute doppler and range steering tensors
    doppler_steering_tensor = tf.math.exp(-1j * tf.expand_dims(targets_omega_d, 1) * tf.cast(tf.expand_dims(tf.range(K), 0), dtype=tf.complex128))
    range_steering_tensor = tf.math.exp(-1j * tf.expand_dims(targets_omega_r, 1) * tf.cast(tf.expand_dims(tf.range(N), 0), dtype=tf.complex128))
    
    # compute range-doppler and signal and get the SCNR
    rd_signal = tf.expand_dims(range_steering_tensor, 2) * tf.expand_dims(doppler_steering_tensor, 1)
    SCNR_db = get_SCNR_db(config, targets_num)
    
    # Adjust phase of the range-Doppler signal based on the configuration
    if config.signal_random_phase:
        rd_signal = rd_signal * tf.math.exp(1j * tf.cast(tf.expand_dims(tf.expand_dims(tf.random.uniform(targets_num, 0, 2 * np.pi), 1), 1), dtype=tf.complex128))
    elif config.signal_physical_phase:
        targets_tau0 = (2 * targets_rng) / 3e8
        rd_signal = rd_signal * tf.expand_dims(tf.expand_dims(tf.math.exp(1j * (tf.cast(-2 * np.pi * config.f_c * targets_tau0 + np.pi * (config.B_chirp / (config.N * config.f_s)) * (targets_tau0 ** 2), dtype=tf.complex128))), 1), 1)

    # compensate for the appropriate SCNR level
    s_norm = tf.math.real(tf.norm(rd_signal, axis=[1, 2]))
    sig_amp = (10 ** (tf.cast(SCNR_db, dtype=tf.float64) / 20.0)) * (tf.cast(cn_norm, dtype=tf.float64) / s_norm)
    rd_signal = tf.reduce_sum(tf.cast(tf.expand_dims(tf.expand_dims(sig_amp, -1), -1), dtype=tf.complex128) * rd_signal, axis=0)
    
    # gen label vector
    trgt_inds_vel = tf.expand_dims(tf.math.argmin(tf.abs(tf.expand_dims(targets_vel, 1) - tf.expand_dims(recon_vec_vel, 0)), axis=1), 1)
    trgt_inds_rng = tf.expand_dims(tf.math.argmin(tf.abs(tf.expand_dims(targets_rng, 1) - tf.expand_dims(recon_vec_rng, 0)), axis=1), 1)
    tf.debugging.assert_type(trgt_inds_vel, tf.int64)
    tf.debugging.assert_type(trgt_inds_rng, tf.int64)
    trgt_inds = tf.concat((trgt_inds_rng, trgt_inds_vel), 1)
    tf.debugging.assert_type(trgt_inds, tf.int64)
    label_tensor = tf.scatter_nd(trgt_inds, tf.squeeze(tf.ones_like(trgt_inds_vel), 1), (recon_vec_rng.shape[0], recon_vec_vel.shape[0]))

    #gen paramter value and SCNR tensor
    param_val_tensor = (tf.concat((targets_rng, tf.ones(config.num_targets - targets_num) * -1000.0), axis=0),
                        tf.concat((targets_vel, tf.ones(config.num_targets - targets_num) * -1000.0), axis=0))
    scnr_tensor = tf.concat((SCNR_db, tf.ones(config.num_targets - targets_num) * -1000.0), axis=0)

    return rd_signal, label_tensor, param_val_tensor, scnr_tensor

#------------------------------------------#

def get_SCNR_db(config, targets_num):

    if config.SCNR_db_random_choice:
        SCNRs_eval = tf.constant(config.SCNRs_eval, dtype=tf.float32)
        scnr_eval_inds = tf.random.uniform(tf.cast(targets_num, dtype=tf.int32), minval=0, maxval=len(config.SCNRs_eval), dtype=tf.int32)
        SCNR_db = tf.gather(SCNRs_eval, scnr_eval_inds)
    elif config.SCNR_db_random_constant:
        SCNRs_eval = tf.constant(config.SCNRs_eval, dtype=tf.float32)
        scnr_eval_inds = tf.random.uniform([1], minval=0, maxval=len(config.SCNRs_eval), dtype=tf.int32)
        SCNR_db = tf.gather(SCNRs_eval, scnr_eval_inds) * tf.ones(targets_num)
    elif config.random_SCNR:
        SCNR_db = tf.random.uniform(targets_num, minval=config.SCNR_db_range[0], maxval=config.SCNR_db_range[1] + 0.001)
    else:
        SCNR_db = tf.constant(config.SCNR_db, dtype=tf.float32) * tf.ones(targets_num)

    return SCNR_db

#------------------------------------------#

def split_auxillary_structure(mat_complex, mat_label, param_val, scnr, gamma_shape, clutter_vel, clutter_label_tensor):
    return tf.squeeze(mat_complex), mat_label, (param_val, scnr, gamma_shape, clutter_vel, clutter_label_tensor)

#------------------------------------------#

def tf_dataset_pipeline(config, data):

    # centering and reshape mat_complex
    def two_stage_fc_preprocess(mat_complex, label):
        mat = cube_center_and_reshape(mat_complex)
        return mat, label

    # standartization: element-wise division by the std 
    def two_stage_fc_stdize(mat_complex, mat_label, aux):
        mat_complex = mat_complex / tf.cast(tf.math.reduce_std(mat_complex, axis=0), dtype=tf.complex128)
        return mat_complex, mat_label, aux
    
    # ????? sends to previous function - why ??????
    def two_stage_fc_preprocess_cg(mat_complex, mat_label, aux):
        mat_complex, mat_label = two_stage_fc_preprocess(mat_complex, mat_label)
        return mat_complex, mat_label, aux

    # transpose
    def transpose_mat_complex(mat_complex, mat_label, aux):
        mat_complex = tf.transpose(mat_complex)
        return mat_complex, mat_label, aux

    # Concatenates the real and imaginary parts
    def concat_real_imag_cg(mat_complex, mat_label, aux):
        return tf.concat((tf.math.real(mat_complex), tf.math.imag(mat_complex)), axis=-1), mat_label, aux

    # summing along a specified axis (reduce_axis) and then clipping the values to be between 0 and 1
    def preprocess_label2d(mat_label):
        return tf.cast(tf.clip_by_value(tf.reduce_sum(mat_label, axis=reduce_axis), 0, 1), dtype=tf.float32)

    # ?
    def cg_preprocess_label_2dims(mat_complex, mat_label, aux):
        mat_label = tf.cast(tf.clip_by_value(tf.reduce_sum(mat_label, axis=reduce_axis), 0, 1), dtype=tf.float32)
        return mat_complex, mat_label, aux

    # squeeze and clip values between 0 and 1
    def cg_preprocess_label_1dim(mat_complex, mat_label, aux):
        mat_label = tf.squeeze(tf.cast(tf.clip_by_value(mat_label, 0, 1), dtype=tf.float32))
        return mat_complex, mat_label, aux

    # centers by subtracting the mean of reshaped input along the last axis and then reshapes it back to the original shape. 
    def cube_center_and_reshape(mat):
        mat_center = mat - tf.reduce_mean(tf.reshape(mat, (-1, mat.shape[-1])), axis=0)
        return tf.reshape(mat_center, (-1, mat_center.shape[-1]))

    '''
    if config.data_name is "ipix" & config.model_name is "Detection-TwoStage-FC":
    if config.estimation_params is ["rng"], the transpose_mat_complex function is applied to the data
    the function is sent to cube_center_and_reshape (from two_stage_fc_preprocess...)
    if config.two_stage_fc_stdize is True - it is divided element-wise by the std
    we then concatenate the real and imaginary parts, The reduce_axis is set to 1 if config.estimation_params is "rng", else 0.
    '''
    if config.data_name == "ipix":
        if config.model_name == "Detection-TwoStage-FC":
            assert config.estimation_params == ["rng"] or config.estimation_params == ["vel"]
            if config.estimation_params == ["rng"]:
                data = data.map(transpose_mat_complex)
            data = data.map(two_stage_fc_preprocess_cg)
            if config.two_stage_fc_stdize:
               data = data.map(two_stage_fc_stdize)
            data = data.map(concat_real_imag_cg)
            reduce_axis = 1 if config.estimation_params == ["rng"] else 0
            data = data.map(cg_preprocess_label_2dims)

    
    #The first is similar to the previous section:
    if config.data_name == "compound_gaussian" or config.data_name == "wgn":
        if config.model_name == "Detection-TwoStage-FC":
            assert config.estimation_params == ["rng"] or config.estimation_params == ["vel"]
            if config.estimation_params == ["rng"]:
                data = data.map(transpose_mat_complex)
            data = data.map(two_stage_fc_preprocess_cg)
            if config.two_stage_fc_stdize:
                data = data.map(two_stage_fc_stdize)
            data = data.map(concat_real_imag_cg)
    # from here comes the difference:
    # if config.compound_gaussian_dims is 2, the reduce_axis is set to 1 if config.estimation_params is ["rng"],else 0, and the cg_preprocess_label_2dims function is applied to the data.
    # if config.compound_gaussian_dims is not 2, we squeeze and clip values between 0 and 1.
    # if config.model_name is "Detection-FC" and the data is "compound_gaussian" or "wgn", the lambda function is applied to the data. 
    # the lambda function concatenates the real and imaginary parts of t along dimension 0, squeezes the label, and performs value clipping between 0 and 1.
            if config.compound_gaussian_dims == 2:
                reduce_axis = 1 if config.estimation_params == ["rng"] else 0
                data = data.map(cg_preprocess_label_2dims)
            else:
                data = data.map(cg_preprocess_label_1dim)
        elif config.model_name == "Detection-FC":
            data = data.map(lambda t, label, aux: (tf.squeeze(tf.concat((tf.math.real(t), tf.math.imag(t)), 0)),
                                                   tf.cast(tf.clip_by_value(tf.squeeze(label), 0, 1), dtype=tf.float32), aux))

    return data