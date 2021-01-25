import numpy as np
import pandas as pd
import scipy as sc
import tqdm as tq
import matplotlib as ma
import seaborn as se
import sklearn as sk
import llvmlite as ll
import numba as nu
import gpustat as gp
import librosa as li
import tensorflow as tf
import tensorflow_addons as tfa
import keras as ke
import classification_models
import lightgbm as lgb
import xgboost as xgb
#import cupy as cp


print('========== testing installed python packages ==========')
print('numpy version: ', np.__version__)
print('pandas version:', pd.__version__)
print('scipy version:', sc.__version__)
print('tqdm version:', tq.__version__)
print('matplotlib version:', ma.__version__)
print('seaborn version:', se.__version__)
print('sklearn version:', sk.__version__)
print('gpustat version:', gp.__version__)
print('llvmlite version:', ll.__version__)
print('numba version:',nu.__version__)
print('librosa version:',li.__version__)
print('tensorflow version: ', tf.__version__)
print('tensorflow_addons version: ', tfa.__version__)
print('keras version:', ke.__version__)
print('image-classifiers version:',classification_models.__version__)
print('lightgbm version:', lgb.__version__)
print('xgboost version:', xgb.__version__)
#print('cupy version:', cp.__version__)
