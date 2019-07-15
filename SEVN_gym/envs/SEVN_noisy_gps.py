from __future__ import print_function, division
import enum
import math
import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from matplotlib import pyplot as plt
import cv2
import gym
import gzip
import time
from gym import spaces
import h5py
import pickle
from SEVN_gym.data import _ROOT
from SEVN_gym.envs.SEVN_base import SEVNBase
from SEVN_gym.envs import utils


class SEVNNoisyGPS(SEVNBase):
    def __init__(self, obs_shape=(4, 84, 84), use_image_obs=True, use_gps_obs=True, use_visible_text_obs=True, use_full=False, reward_type=None, noise_scale=0.0):
        super(SEVNNoisyGPS, self).__init__(obs_shape, use_image_obs, use_gps_obs, use_visible_text_obs, use_full, reward_type)
        self.noise_scale = noise_scale

    def sample_gps(self, groundtruth):
        coords = groundtruth[['x', 'y']]
        x_scale = self.meta_df.x.max() - self.meta_df.x.min()
        y_scale = self.meta_df.y.max() - self.meta_df.y.min()
        x = (coords.at[0, 'x'] + np.random.normal(loc=0.0, scale=self.noise_scale)) / x_scale
        y = (coords.at[0, 'y'] + np.random.normal(loc=0.0, scale=self.noise_scale)) / y_scale
        return (x, y)
