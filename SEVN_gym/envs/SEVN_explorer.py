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


class SEVNExplorer(SEVNBase):
    def __init__(self, obs_shape=(4, 84, 84), use_image_obs=True, use_gps_obs=False, use_visible_text_obs=True, use_full=False, reward_type=None):
        super(SEVNExplorer, self).__init__(obs_shape, use_image_obs, False, use_visible_text_obs, use_full, reward_type)
        self.max_num_steps = 300
        self.seen_house_nums = []

    def step(self, a):
        done = False
        was_successful_trajectory = False
        oracle = False

        reward = 0.0
        self.num_steps_taken += 1
        action = self._action_set(a)
        image, x, w = self._get_image()
        visible_text = self.get_visible_text(x, w)

        if self.num_steps_taken >= self.max_num_steps and done == False:
            done = True
        elif action == self.Actions.FORWARD:
            self.transition()
        else:
            self.turn(action)
        reward = self.compute_reward(x, {'visible_text':visible_text}, done)
        obs = {"image": image, "visible_text": visible_text}
        obs = self.obs_wrap(obs)

        info = {}
        if done:
            self.needs_reset = True

        return obs, reward, done, info

    def reset(self):
        self.needs_reset = False
        self.num_steps_taken = 0
        self.seen_house_nums = []
        image, x, w = self._get_image()
        obs = {"image": image, "visible_text": self.get_visible_text(x, w)}
        obs = self.obs_wrap(obs)
        return obs

    def compute_reward(self, x, info, done):
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on an a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in info and compute it accordingly.

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information

        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:

                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        """
        reward = 0
        house_numbers = SEVNExplorer.convert_house_vec_to_ints(info['visible_text']['house_numbers'])
        for num in house_numbers:
            if num not in self.seen_house_nums:
                reward += 1
                self.seen_house_nums.append(num)
        return reward

    def obs_wrap(self, obs):
        coord_holder = np.zeros((1, 84, 84), dtype=np.float32)

        if self.use_gps_obs:
            coord_holder[0, 0, :4] = obs['rel_gps']

        if self.use_visible_text_obs:
            coord_holder[0, 1, :2 * self.num_streets] = obs['visible_text']['street_names']
            coord_holder[0, 2, :] = obs['visible_text']['house_numbers'][:84]
            coord_holder[0, 3, :36] = obs['visible_text']['house_numbers'][84:120]
        if not self.use_image_obs:
            obs['image'] = np.zeros((3, 84, 84))

        coord_holder[0, 4, :40] = np.zeros(coord_holder[0, 4, :40].shape)
        coord_holder[0, 4, 40:40 + self.num_streets] = np.zeros(coord_holder[0, 4, 40:40 + self.num_streets].shape)

        out = np.concatenate((obs['image'], coord_holder), axis=0)
        return out
