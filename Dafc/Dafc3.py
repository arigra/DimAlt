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


class TwoStageFcLayer(tf.keras.layers.Layer):
    """
    Perform TwoStage Fully-Connected
    input tensor: (input_row_dim, input_col_dim)
    output tensor: (row_units, column_units)
    """
    def __init__(self, row_units, column_units, l2_lamda, activation_name, use_batchnorm, dropout_rate, is_first, **kwargs):
        super(TwoStageFcLayer, self).__init__()

        self.row_units = row_units
        self.column_units = column_units
        self.l2_lamda = l2_lamda
        self.activation_name = activation_name
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

        if l2_lamda > 0.0:
            # kernel_reg = EyeRegularizer() if is_first else tf.keras.regularizers.l2(l2_lamda)
            self.col_dense = tf.keras.layers.Dense(column_units,
                                bias_regularizer=tf.keras.regularizers.l2(l2_lamda),
                                kernel_regularizer=tf.keras.regularizers.l2(l2_lamda))
            # kernel_regularizer=kernel_reg)
            self.row_dense = tf.keras.layers.Dense(row_units,
                                bias_regularizer=tf.keras.regularizers.l2(l2_lamda),
                                kernel_regularizer=tf.keras.regularizers.l2(l2_lamda))
        else:
            self.col_dense = tf.keras.layers.Dense(column_units)
            self.row_dense = tf.keras.layers.Dense(row_units)
        self.col_activation = Activation(activation_name)
        self.row_activation = Activation(activation_name)

        if self.use_batchnorm:
            self.col_bnorm = tf.keras.layers.BatchNormalization()
            self.row_bnorm = tf.keras.layers.BatchNormalization()

        if self.dropout_rate > 0.0:
            self.col_dropout = Dropout(rate=self.dropout_rate)
            self.row_dropout = Dropout(rate=self.dropout_rate)
        return

    def call(self, input_tensor, training=False):
        # input_tensor: [batch, input_row_dim, input_col_dim]

        x = self.col_dense(input_tensor) # [batch, input_row_dim, column_units]
        x = self.col_activation(x)
        if self.use_batchnorm:
            x = self.col_bnorm(x, training=training)
        if self.dropout_rate > 0.0:
            x = self.col_dropout(x, training=training)

        x = tf.transpose(x, perm=[0, 2, 1]) # [batch, column_units, input_row_dim]

        x = self.row_dense(x) # [batch, column_units, row_units]
        x = self.row_activation(x)
        if self.use_batchnorm:
            x = self.row_bnorm(x, training=training)
        if self.dropout_rate > 0.0:
            x = self.row_dropout(x, training=training)

        x = tf.transpose(x, perm=[0, 2, 1]) # [batch, row_units, col_units]

        return x


#-----------------------------#

def TwoStageFCModel(config, include_top=True, name_str=""):
    l2_lamda = config.l2_reg_parameter
    two_stage_fc_dims = config.two_stage_fc_dims
    two_stage_fc_use_batch_norm = config.two_stage_fc_use_batch_norm
    two_stage_fc_dropout_rate = config.two_stage_fc_dropout_rate
    activation_name = config.activation
    input_layer = Input(shape=config.model_input_dim, name="input")
    x = input_layer

    for i in range(len(two_stage_fc_dims)):
        is_first = True if i==0 else False
        x = TwoStageFcLayer(row_units=two_stage_fc_dims[i][0], column_units=two_stage_fc_dims[i][1], l2_lamda=l2_lamda,
                             activation_name=activation_name,
                             use_batchnorm=two_stage_fc_use_batch_norm[i], dropout_rate=two_stage_fc_dropout_rate[i], is_first=is_first)(x)
    if config.two_stage_fc_use_gap:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    else:
        x = Flatten()(x)

    config.dense_sizes = config.two_stage_fc_dense_sizes
    config.dense_dropout = config.two_stage_fc_dense_dropout
    config.fc_batchnorm = config.two_stage_fc_dense_batchnorm
    if config.dense_sizes != []:
        x = FCSkeletonModel(config, x, create_model=False)

    if config.point_cloud_reconstruction:
        last_layer_dim = config.model_output_dim[0]
    elif config.mode == "Estimation":
        last_layer_dim = len(config.estimation_params)
    elif config.mode == "Detection":
        last_layer_dim = 2
    else:
        raise Exception("TwoStageFCModel(): Unsupported config.mode")

    if l2_lamda > 0:
        y_hat = Dense(last_layer_dim,
                  kernel_regularizer=tf.keras.regularizers.l2(l2_lamda),
                  bias_regularizer=tf.keras.regularizers.l2(l2_lamda))(x)
    else:
        y_hat = Dense(last_layer_dim)(x)

    if config.point_cloud_reconstruction:
        y_hat = activation(config, 'sigmoid', y_hat)
    elif config.mode == "Detection":
        y_hat = activation(config, 'softmax', y_hat)

    model = tf.keras.Model(inputs=input_layer, outputs=y_hat, name="TwoStageFcModel")

    return model

#-----------------------------#


def activation(config, activation_name, x):
    if activation_name == 'leaky_relu':
        return LeakyReLU(alpha=config.leaky_alpha)(x)
    else:
        return Activation(activation_name)(x)

#-----------------------------#

def FCSkeletonModel(config, input_layer, create_model=True, output_name="", dense_activations=None):

    # get parameters from config file
    dense_sizes = config.dense_sizes
    dense_dropout = [0 for _ in range(len(dense_sizes))] if config.dense_dropout is None else config.dense_dropout
    activation_name = config.activation
    if dense_activations == None:
        dense_activations = [activation_name for j in range(len(dense_sizes))]
    else:
        assert len(dense_activations) == len(dense_sizes)
    l2_lamda = config.l2_reg_parameter

    x = input_layer
    # Dense
    for i, size in enumerate(dense_sizes):
        if l2_lamda != 0:
            x = Dense(size, kernel_regularizer=tf.keras.regularizers.l2(l2_lamda),
                      bias_regularizer=tf.keras.regularizers.l2(l2_lamda))(x)
        else:
            x = Dense(size)(x)
        if dense_activations[i] != "None":
            x = activation(config, dense_activations[i], x)
        if config.fc_batchnorm:
            x = BatchNormalization()(x)
        # Dropout, at the last layer apply dropout after flatten
        if dense_dropout[i] != 0:
            x = Dropout(rate=dense_dropout[i])(x)

    output_layer = x
    if create_model:
        model = Model(input_layer, output_layer)
        if output_name!= "":
            model.layers[len(model.layers) - 1]._name = output_name
        return model
    else:
        return output_layer

#-----------------------------#

def EstimationFCModel(config):
    input_layer = Input(shape=config.model_input_dim)
    l2_lamda = config.l2_reg_parameter

    model_fc_skeleton = FCSkeletonModel(config, input_layer)
    x = model_fc_skeleton.output

    # prediction neuron
    output_layer = Dense(len(config.estimation_params),
                         kernel_regularizer=tf.keras.regularizers.l2(l2_lamda),
                         bias_regularizer=tf.keras.regularizers.l2(l2_lamda))(x)

    model = tf.keras.Model(input_layer, output_layer, name="EstimationFCModel")

    return model

#-----------------------------#


def build_model(config):
    if config.load_complete_model:
        model = tf.keras.models.load_model(config.load_model_path, compile=False)
        print("\n" + "!" * 25 + "\n" + "WARNING: LOADING MODEL FROM: {}".format(config.load_model_path) + "\n" + "!" * 25 + "\n")
    elif config.model_name == "Estimation-FC":
        model = EstimationFCModel(config)
    elif config.model_name == "Estimation-TwoStage-FC" or config.model_name == "Detection-TwoStage-FC":
         model = TwoStageFCModel(config)
    else:
        raise ValueError("'{}' is an invalid model name")

    model.summary(line_length=140)
    if config.load_complete_model:
        print("\n" + "!" * 25 + "\n" + "WARNING: LOADING MODEL FROM: {}".format(config.load_model_path) + "\n" + "!" * 25 + "\n")
    try:
        model_plot_dir = os.path.join(config.tensor_board_dir, "model_plot")
        if not os.path.exists(model_plot_dir):
            os.makedirs(model_plot_dir)
        img_pth = os.path.join(model_plot_dir, config.model_name + ".png")
        plot_model(model, img_pth, show_shapes=False)
        print("saved model plot at: {}".format(img_pth))
    except Exception as e:
        print("Failed to plot model:" + str(e))

    return model
