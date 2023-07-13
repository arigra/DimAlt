import os
import sys
import glob
import scipy
import shutil
import pickle
import mlflow
import logging
import datetime
import argparse
import commentjson
import numpy as np
from bunch import Bunch
import tensorflow as tf
from pathlib import Path
from random import randint
from collections import OrderedDict
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Input
from tensorflow.keras.layers import Dropout,  Activation, LeakyReLU, AveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, Callback, EarlyStopping

#--------------------------------------------------#

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=5):
        super().__init__()
        self.gamma = gamma
        return

    def call(self, y_true, y_pred):
        _nll2 = tf.keras.losses.binary_crossentropy(tf.expand_dims(y_true, -1), tf.expand_dims(y_pred, -1))
        pt = tf.zeros_like(_nll2)
        # pt for y_true == 1
        ind1 = tf.where(y_true >= 0.999)
        pt1 = tf.gather_nd(y_pred, ind1)
        pt = tf.tensor_scatter_nd_update(pt, ind1, pt1)
        # pt for y_true == 0
        ind0 = tf.where(y_true < 0.999)
        pt0 = 1 - tf.gather_nd(y_pred, ind0)
        pt = tf.tensor_scatter_nd_update(pt, ind0, pt0)
        # compute Focal Loss
        loss = tf.math.pow(1 - pt, self.gamma) * _nll2

        return tf.reduce_mean(loss, -1)

#--------------------------------------------------#

class ClassBalancedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, e1, e0, predefined_weight, use_penalize_margin=False, penalize_margin=8, balanced_loss_beta=0.99, recon_dim=128):
        super().__init__()
        self.balanced_loss_beta = balanced_loss_beta
        self.e1 = e1
        self.e0 = e0
        self.predefined_weight = predefined_weight
        self.use_penalize_margin = use_penalize_margin
        self.penalize_margin = penalize_margin

        if self.predefined_weight > 0.0:
            self.weight0 = 1.0
            self.weight1 = self.predefined_weight
        else:
            self.weight0 = ((1.0 - self.balanced_loss_beta) / (1.0 - self.balanced_loss_beta ** self.e0))
            self.weight1 = ((1.0 - self.balanced_loss_beta) / (1.0 - self.balanced_loss_beta ** self.e1))

        if self.use_penalize_margin:
            assert self.penalize_margin > 1
            self.ind1_getter = self.get_ind1_with_margin
        else:
            self.ind1_getter = self.get_ind1

        self.label_map = lambda y_true: y_true

        self.recon_dim = recon_dim


    def get_ind1(self, y_true):
        ind1 = tf.where(y_true >= 0.9999)
        return ind1

    def get_ind1_with_margin(self, y_true):
        def get_ind_margin_inds(t):
            # t = tf.where(y_true >= 0.9999)[0]
            ind_plus = tf.concat((tf.expand_dims(tf.gather(t, 0) * tf.cast(tf.ones(self.penalize_margin), dtype=tf.int64), 1),
                 tf.expand_dims(tf.minimum(tf.gather(t, 1) + tf.range(self.penalize_margin, dtype=tf.int64), self.recon_dim - 1), 1)), axis=1)
            ind_minus = tf.concat((tf.expand_dims(tf.gather(t, 0) * tf.cast(tf.ones(self.penalize_margin - 1), dtype=tf.int64), 1),
                 tf.expand_dims(tf.maximum(tf.gather(t, 1) - tf.range(1, self.penalize_margin, dtype=tf.int64), 0), 1)), axis=1)

            return tf.concat((ind_plus, ind_minus), axis=0)
        ind1 = tf.reshape(tf.map_fn(get_ind_margin_inds, tf.where(y_true > 0.9999)),(-1, 2))
        return ind1

    def call(self, y_true, y_pred):
        # call function for the regular single-label tensor
        # y_true = [batch_dim, recon_dim], y_pred = [batch_dim, recon_dim]
        ind1 = self.ind1_getter(y_true)
        # y_true = self.label_map(y_true)
        _nll2 = tf.keras.losses.binary_crossentropy(tf.expand_dims(self.label_map(y_true), -1), tf.expand_dims(y_pred, -1))
        _nll_subset = self.weight1 * tf.gather_nd(_nll2, ind1)
        _nll2 = tf.tensor_scatter_nd_update(_nll2, ind1, _nll_subset)

        return tf.reduce_mean(_nll2, -1)

#--------------------------------------------------#

def CBBCE_get_n0_n1(config, data):
    if len(data['train'].element_spec[1].shape) == 1:
        n_total = len(data['train']) * data['train'].element_spec[1].shape[0]
        n1 = sum([len(np.where(y >= 0.9999)[0]) for X, y, aux in data['train'].as_numpy_iterator()])
        n0 = n_total - n1
        return n0, n1, n_total, None

    else:
        n_total = data['train'].element_spec[1].shape[0] * data['train'].element_spec[1].shape[1] * len(data['train'])
        if len(list(data['train'].element_spec[1].shape)) == 3:
            n1 = sum([len(np.where(y[:, :, 0] >= 0.9999)[0]) for X, y, aux in data['train'].as_numpy_iterator()])
        else:
            n1 = sum([len(np.where(y >= 0.9999)[0]) for X, y, aux in data['train'].as_numpy_iterator()])
        n0 = n_total - n1

    return n0, n1, n_total, None

#--------------------------------------------------#

class KerasTrainer(object):
    """
    General Keras Trainer class
    """

    def __init__(self, model, data, config, sweep_string=None):
        self.model_train = model
        # self.model_eval = model['eval']
        self.data = data
        self.config = config
        self.exp_name_time = config.exp_name_time
        self.sweep_string = sweep_string
        self.callback_list = []
        self.optimizer = self.get_optimizer(config.optimizer)

    def get_optimizer(self, name):
        if name == "adam":
            return Adam(learning_rate=self.config.learning_rate)
        else:
            raise Exception('Unsupported optimizer !!')

    def add_callbacks(self):
        self.callback_list = []
        checkpoint_dir = os.path.join(self.config.tensor_board_dir, 'checkpoints')
        checkpoint_best_filepath = os.path.join(checkpoint_dir, 'model_checkpoint_best')
        checkpoint_epoch_filepath = os.path.join(checkpoint_dir, 'model_checkpoint_epoch')
        # Save best Epoch model
        if self.config.use_model_checkpoint_best:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # save best model
            save_model_best_callback = ModelCheckpoint(filepath=checkpoint_best_filepath, save_weights_only=False,
                                                       monitor=self.config.model_checkpoint_best_metric,
                                                       save_best_only=True, verbose=1)
            self.callback_list.append(save_model_best_callback)

        # Save model periodically
        if self.config.model_checkpoint_epoch_period > 0:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # save model periodically
            save_model_epoch_callback = ModelCheckpoint(filepath=checkpoint_epoch_filepath, save_weights_only=False,
                                                        save_best_only=False, verbose=1)
            self.callback_list.append(save_model_epoch_callback)

        # CSV Logger for per-epoch logging
        if self.config.save_fit_history:
            fit_log_dir = os.path.join(self.config.tensor_board_dir, 'fit_log')
            if not os.path.exists(fit_log_dir):
                os.makedirs(fit_log_dir)
            if self.sweep_string is None:
                csv_logger_path = os.path.join(fit_log_dir, '{}_fit_log.csv'.format(self.config.exp_name_time))
            else:
                csv_logger_path = os.path.join(fit_log_dir,
                                               '{}_fit_log.csv'.format(self.sweep_string))
            csv_logger = CSVLogger(csv_logger_path)
            self.callback_list.append(csv_logger)

        # Early-Stopping
        if self.config.use_early_stop:
            # patiennce_epochs = int(self.config.early_stop_patience*self.config.num_epochs) if self.config.early_stop_patience<=1 else self.config.early_stop_patience
            # early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor=self.config.early_stop_metric, verbose=1, patience=patiennce_epochs)
            early_stop_callback = EarlyStoppingCallback(self.config)
            self.callback_list.append(early_stop_callback)

        if self.config.use_lr_scheduler:
            assert not self.config.use_lr_scheduler_deriv
            lr_scheduler = LrScheduler(self.config)
            lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler.schedule)
            self.callback_list.append(lr_scheduler_callback)
        if self.config.use_lr_scheduler_plateau:
            assert not self.config.use_lr_scheduler_deriv
            assert not self.config.use_lr_scheduler
            lr_scheduler_plateau = LrSchedulerPlateau(self.config)
            self.callback_list.append(lr_scheduler_plateau)

        if self.config.use_lr_scheduler_deriv:
            assert not self.config.use_lr_scheduler
            lr_scheduler_callback = LrSchedulerDeriv(self.config)
            self.callback_list.append(lr_scheduler_callback)

        if self.config.stop_max_acc:
            callback_obj = StoppingAtMaxAccuracy()
            self.callback_list.append(callback_obj)

        self.callback_list = None if self.callback_list == [] else self.callback_list
        return

    def compile(self, loss_fn, metrics):
        # compile the model
        self.model_train.compile(optimizer=self.optimizer, loss=loss_fn, metrics=metrics)

    def train(self):
        # # model checkpoints for
        # self.add_callbacks()

        # train the model
        history = self.model_train.fit(self.data['train'], epochs=self.config.num_epochs,
                                       validation_data=self.data['valid'],
                                       callbacks=self.callback_list,
                                       verbose=self.config.fit_verbose)
        return history


#--------------------------------------------------#

def positive_binary_cross_entropy(y_true, y_pred):
    ind1 = tf.where(y_true >= 0.9999)
    value = tf.cond(tf.size(ind1)==0, lambda: tf.constant(0.0),
            lambda :tf.reduce_mean(metrics.binary_crossentropy(tf.expand_dims(tf.gather_nd(y_true, ind1), -1),
                                               tf.expand_dims(tf.gather_nd(y_pred, ind1), -1)), axis=-1))
    return value
def negative_binary_cross_entropy(y_true, y_pred):
    ind0 = tf.where(y_true < 0.9999)
    value = tf.reduce_mean(metrics.binary_crossentropy(tf.expand_dims(tf.gather_nd(y_true, ind0), -1),
                                                       tf.expand_dims(tf.gather_nd(y_pred, ind0), -1)), axis=-1)
    return value


#--------------------------------------------------#

class ClassificationTrainerKeras(KerasTrainer):

    def __init__(self, model, data, config, sweep_string=None):
        super().__init__(model, data, config, sweep_string)
        # tf.config.experimental_run_functions_eagerly(True)
        assert config.data_name == "compound_gaussian" or config.data_name == "ipix" or config.data_name == "wgn"
        self.sweep_string = '' if sweep_string is None else sweep_string
        tune_hist_dir = os.path.join(self.config.tensor_board_dir, 'tune_hist')
        if not os.path.exists(tune_hist_dir):
            os.makedirs(tune_hist_dir)
        self.tune_hist_path = os.path.join(tune_hist_dir, 'tune_hist_' + self.sweep_string + '.csv')
        assert config.point_cloud_reconstruction
        if self.config.use_CBBCE:
            n0, n1, n_total, n2 = CBBCE_get_n0_n1(config, data)
            self.loss_fn = ClassBalancedBinaryCrossEntropy(e0=n0 / n_total, e1=n1 / n_total,
                predefined_weight=self.config.CBBCE_predefined_weight,
                use_penalize_margin=self.config.CBBCE_use_penalize_margin, penalize_margin=self.config.CBBCE_penalize_margin,
                recon_dim=self.config.model_output_dim[0])
        else:
            self.loss_fn = FocalLoss()

        self.metrics = [metrics.binary_crossentropy, positive_binary_cross_entropy, negative_binary_cross_entropy,
                    metrics.AUC(name='auc'), 'accuracy', metrics.FalsePositives(name="fp"), metrics.TruePositives(name="tp")]

    def train(self):
        # super().compile(self.loss_fn, self.metrics)
        self.model_train.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)
        super().add_callbacks()

        # train the model
        data_train = self.data['train']
        data_valid = self.data['valid']

        if self.config.data_name == "compound_gaussian" or self.config.data_name == "ipix" or self.config.data_name == "wgn":
            data_train = data_train.map(compound_gaussian_split_aux_trainer)
            data_valid = data_valid.map(compound_gaussian_split_aux_trainer)

        history = self.model_train.fit(data_train, epochs=self.config.num_epochs, validation_data=data_valid, callbacks=self.callback_list, verbose=self.config.fit_verbose)
        return history


    def evaluate(self):
        eval_res = self.model_train.evaluate(self.data['valid'])
        return eval_res

    def test(self):
        test_res = self.model_train.evaluate(self.data['test'])
        return test_res

    def train_eval(self):
        return self.model_train.evaluate(self.data['train'])

    def predict(self, X):
        return self.model_train.predict(X)


#--------------------------------------------------#

def build_trainer(model, data, config, sweep_string=None):
    if config.trainer_name == "detection_classification":
        trainer = ClassificationTrainerKeras(model, data, config, sweep_string)
    else:
        raise ValueError("'{}' is an invalid model name")

    return trainer


#--------------------------------------------------#

class LrSchedulerPlateau(Callback):
    def __init__(self, config):
        super(LrSchedulerPlateau, self).__init__()
        self.decay = config.lr_scheduler_plateau_decay
        self.ma_window = config.lr_scheduler_plateau_window + 1
        self.cooldown = config.lr_scheduler_plateau_cooldown
        self.epoch_threshold = int(config.lr_scheduler_plateau_epoch_threshold * config.num_epochs)
        self.val_loss_buffer = []
        self.last_update_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # first epoch is 0
        val_loss = logs.get("val_loss")
        self.val_loss_buffer.append(val_loss)

        if epoch > self.epoch_threshold and epoch > self.ma_window + 1:
            val_loss_mean = np.mean(self.val_loss_buffer[-self.ma_window:-1])
            if (val_loss - val_loss_mean > 1e-4) and (epoch - self.last_update_epoch > self.cooldown) :
                self.model.optimizer.learning_rate = self.model.optimizer.learning_rate * self.decay
                self.last_update_epoch = epoch

        return

#--------------------------------------------------#

class LrScheduler(object):
    def __init__(self, config):
        self.decay = config.lr_scheduler_decay
        self.period = config.lr_scheduler_period
        self.epoch_threshold = int(config.lr_scheduler_epoch_threshold * config.num_epochs)

    def schedule(self, epoch, lr):
        if epoch < self.epoch_threshold:
            return lr
        else:
            if epoch % self.period == 0:
                return self.decay * lr
            else:
                return lr

class EarlyStoppingCallback(Callback):
    def __init__(self, config):
        super(EarlyStoppingCallback, self).__init__()
        self.metric = config.early_stop_metric
        self.epoch_patience = int(config.early_stop_patience*config.num_epochs) if config.early_stop_patience<=1 else config.early_stop_patience
        self.val_loss_buffer = []
        self.stopped_epoch = None

        return
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        self.val_loss_buffer.append(val_loss)

        if epoch > self.epoch_patience + 1:
            val_loss_diff = self.val_loss_buffer[-1] - self.val_loss_buffer[-self.epoch_patience]
            if val_loss_diff > 1e-4:
                self.stopped_epoch = epoch
                self.model.stop_training = True

        return

    def on_train_end(self, logs=None):
        if self.stopped_epoch is not None:
            print("EarlyStoppingCallback(): Epoch %05d: early stopping" % (self.stopped_epoch + 1))
        return

def compound_gaussian_split_aux_trainer(mat, label, aux):
    return mat, label
