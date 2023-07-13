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


#-----------------------------#
# Define Config Class
logger = logging.getLogger("logger")
CONFIG_VERBOSE_WAIVER = ['save_model', 'tracking_uri', 'quiet', 'sim_dir', 'train_writer', 'test_writer', 'valid_writer']
class Config(Bunch):
    """ class for handling dicrionary as class attributes """

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def print(self):
        line_len = 122
        line = "-" * line_len
        logger.info(line + "\n" +
              "| {:^35s} | {:^80} |\n".format('Feature', 'Value') +
              "=" * line_len)
        for key, val in sorted(self.items(), key= lambda x: x[0]):
            if isinstance(val, OrderedDict):
                raise NotImplementedError("Nested configs are not implemented")
            else:
                if key not in CONFIG_VERBOSE_WAIVER:
                    logger.info("| {:35s} | {:80} |\n".format(key, str(val)) + line)
        logger.info("\n")

#-----------------------------#
# Define arguments
def get_args(argv):
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', default=None, type=str, help='path to config file')
    argparser.add_argument('--seed', default=None, type=int, help='randomization seed')
    argparser.add_argument('--exp_name', default=None, type=int, help='Experiment name')
    argparser.add_argument('--num_targets', default=None, type=int, help='Number of simulated targets')
    argparser.set_defaults(quiet=False)
    args, unknown = argparser.parse_known_args(argv)
    #args = argparser.parse_args()

    return args

#-----------------------------#
#read config file
def read_json_to_dict(fname):
    """ read json config file into ordered-dict """
    fname = Path(fname)
    with fname.open('rt') as handle:
        config_dict = commentjson.load(handle, object_hook=OrderedDict)
        return config_dict

def read_config(args):
    """ read config from json file and update by the command line arguments """
    if args.config is not None:
        json_file = args.config
    else:
        json_file = "/Users/arigra/Desktop/DL projects/Dafc/config.json"  # Replace with your default config file path

    config_dict = read_json_to_dict(json_file)
    config = Config(config_dict)

    for arg in sorted(vars(args)):
        key = arg
        val = getattr(args, arg)
        if val is not None:
            setattr(config, key, val)

    if args.seed is None and config.seed is None:
        
        MAX_SEED = sys.maxsize
        config.seed = randint(0, MAX_SEED)

    return config

#-----------------------------#
#GPU Initialization
def gpu_init():
    """ Allows GPU memory growth """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    logger.info("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logger.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.info("MESSAGE", e)

#-----------------------------#
#Logger and Tracker
def set_logger_and_tracker(config):
    ''' configure the mlflow tracker:
        1. set tracking location (uri)
        2. configure exp name/id
        3. define parameters to be documented
    '''

    config.exp_name_time = "{}_{}_{}".format(config.exp_name,datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),config.seed)
    config.tensor_board_dir = os.path.join('..',
                                           'results',
                                           config.exp_name,
                                           config.exp_name_time)

    if not os.path.exists(config.tensor_board_dir):
        os.makedirs(config.tensor_board_dir)

def save_scripts(config,SRC_DIR):
    path = os.path.join(config.tensor_board_dir, 'scripts')
    if not os.path.exists(path):
        os.makedirs(path)
    scripts_to_save = glob.glob('{}/**/*.py'.format(SRC_DIR), recursive=True) + [config.config]
    scripts_to_save = [script for script in scripts_to_save if '{}/results'.format(SRC_DIR) not in script]
    if scripts_to_save is not None:
        for script in scripts_to_save:
            dst_file = os.path.join(path, os.path.basename(script))
            try:
                shutil.copyfile(os.path.join(os.path.dirname(sys.argv[0]), script), dst_file)
            except:
                print()

def print_config(config):
    print('')
    print('#' * 70)
    print('Configurations at beginning of run')
    print('#' * 70)
    for key in config.keys():
        print('{}, {}'.format(key,config['{}'.format(key)]))
    print('')
    print('')