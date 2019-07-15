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

ACTION_MEANING = {
    0: 'LEFT_BIG',
    1: 'LEFT_SMALL',
    2: 'FORWARD',
    3: 'RIGHT_SMALL',
    4: 'RIGHT_BIG',
    5: 'DONE',
    6: 'NOOP',
}

class SEVNDecreasingReward(SEVNBase):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, obs_shape=(4, 84, 84), use_image_obs=True, use_gps_obs=False, use_visible_text_obs=True, use_full=False, reward_type=None):
        self.prev_spl = 100000
        super(SEVNDecreasingReward, self).__init__(obs_shape, use_image_obs, False, use_visible_text_obs, use_full, reward_type)

    def compute_reward(self, x, info, done):
        cur_spl = len(self.shortest_path_length())
        if done and self.is_successful_trajectory(x):
            reward = 2.0
        elif done and not self.is_successful_trajectory(x):
            reward = -2.0
        elif self.prev_spl > cur_spl:
            self.prev_spl = cur_spl
            reward = 1
        else:
            reward = 0
        return reward
