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
from hyrule_gym.data import _ROOT
from hyrule_gym.envs.hyrule_base import HyruleBase, ACTION_MEANING


class HyrulePlay(HyruleBase):

    class Actions(enum.IntEnum):
        LEFT_BIG = 0
        LEFT_SMALL = 1
        FORWARD = 2
        RIGHT_SMALL = 3
        RIGHT_BIG = 4
        DONE = 5
        NOOP = 6

    def __init__(self, obs_shape=(84, 84, 3), use_image_obs=True, use_gps_obs=False, use_visible_text_obs=True, use_full=False, reward_type=None):
        super(HyrulePlay, self).__init__(obs_shape, use_image_obs, use_gps_obs, use_visible_text_obs, use_full, reward_type)
        self.max_num_steps = 100000 
        self._action_set = HyrulePlay.Actions


    def step(self, a):
        done = False
        reward = 0.0
        action = self._action_set(a)
        image, x, w = self._get_image()
        visible_text = self.get_visible_text(x, w)

        if not action == self.Actions.NOOP:
            for text in visible_text['house_numbers']:
                if type(text) == str: print('House number: ' + text)
            for text in visible_text['street_names']:
                if type(text) == str: print('Street name: ' + text)

        if action == self.Actions.FORWARD:
            self.transition()
        elif action == self.Actions.DONE:
            done = True
            reward = self.compute_reward(x, {}, done)
            print("Total Mission Reward: " + str(reward))
        else:
            self.turn(action)

        reward = self.compute_reward(x, {}, done)
        
        obs = {"image": image, "mission": self.goal_address, "visible_text": visible_text}
        self.num_steps_taken += 1
        s = "obs: "
        for k, v in obs.items():
            if k != "image":
                s = s + ", " + str(k) + ": " + str(v)
        return obs, reward, done, {}

    def render(self, mode='human'):
        img, x, w = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def _get_image(self, high_res=False, plot=False):
        # TODO: High res env variant
        # if high_res:
        #     path = "data/data/mini-corl/panos/pano_"+ str(int(30*self.G.nodes[self.agent_loc]['timestamp'])).zfill(6) + ".png"
        #     img = cv2.resize(cv2.imread(path)[:, :, ::-1], (960, 480))
        #     obs_shape = (480, 480, 3)
        
        img = self.images_df[self.meta_df.loc[self.agent_loc, 'frame'][0]]
        obs_shape = self.observation_space.shape

        pano_rotation = self.norm_angle(self.meta_df.loc[self.agent_loc, 'angle'][0] + 90)
        w = obs_shape[0]
        y = img.shape[0] - obs_shape[0]
        h = obs_shape[0]
        x = int((self.norm_angle(-self.agent_dir + pano_rotation) + 180)/360 * img.shape[1])

        if (x + w) % img.shape[1] != (x + w):
            res_img = np.zeros(obs_shape)
            offset = img.shape[1] - (x % img.shape[1])
            res_img[:, :offset] = img[y:y+h, x:x + offset]
            res_img[:, offset:] = img[y:y+h, :(x + w) % img.shape[1]]
        else:
            res_img = img[:, x:x + w]

        if plot:
            _, ax = plt.subplots(figsize=(18, 18))
            ax.imshow(res_img.astype(int))
            plt.show()
        return res_img, x, w

    def reset(self):
        self.num_steps_taken = 0
        self.goal_idx, self.goal_address, self.goal_dir = self.select_goal(same_segment=True, difficulty=0)
        self.prev_spl = len(self.shortest_path_length())
        self.start_spl = self.prev_spl
        self.agent_gps = self.sample_gps(self.meta_df.loc[self.agent_loc])
        self.target_gps = self.sample_gps(self.meta_df.loc[self.goal_idx])
        image, x, w = self._get_image()
        rel_gps = [self.target_gps[0] - self.agent_gps[0], self.target_gps[1] - self.agent_gps[1]]
        return {"image": image, "mission": self.goal_address, "rel_gps": rel_gps, "visible_text": self.get_visible_text(x, w)}

    def compute_reward(self, x, info, done):
        cur_spl = len(self.shortest_path_length())
        if done and self.is_successful_trajectory(x):
            reward = 2.0
        elif done and not self.is_successful_trajectory(x):
            reward = -2.0
        elif self.prev_spl - cur_spl > 0:
            reward = 1 
        elif self.prev_spl - cur_spl <= 0:
            reward = -1
        else:
            reward = 0.0
        self.prev_spl = cur_spl
        return reward

    def select_goal(self, same_segment=True, difficulty=0):
        goals = self.meta_df[self.meta_df.is_goal == True]
        G = self.G.copy()
        if same_segment:
            frames = self.meta_df[(self.meta_df.type == "street_segment") & self.meta_df.frame.isin(goals.frame)].frame
            goals_on_street_segment = goals[goals.frame.isin(frames)]
            goal = goals_on_street_segment.loc[np.random.choice(goals_on_street_segment.frame.values.tolist())]
            segment_group = self.meta_df[self.meta_df.frame == goal.frame.iloc[0]].group.iloc[0]
            segment_panos = self.meta_df[(self.meta_df.group == segment_group) & (self.meta_df.type == "street_segment")]
            G.remove_nodes_from(self.meta_df[~self.meta_df.index.isin(segment_panos.index)].index)
        else:
            goal = goals.loc[np.random.choice(goals.frame.values.tolist())]

        goal_idx = self.meta_df[self.meta_df.frame == goal.frame.iloc[0]].frame.iloc[0]
        self.goal_id = goal.house_number
        label = self.meta_df[self.meta_df.frame == int(self.meta_df.loc[goal_idx].frame.iloc[0])]
        label = label[label.is_goal]
        pano_rotation = self.norm_angle(self.meta_df.loc[goal_idx].angle.iloc[0])
        label_dir = self.norm_angle(360 * (label.x_min.values[0] + label.x_max.values[0]) / 2 / 224)
        goal_dir = self.norm_angle(-label_dir + pano_rotation)
        self.agent_dir = 22.5 * np.random.choice(range(-8, 8))
        self.agent_loc =  np.random.choice(segment_panos.frame.unique())

        goal_address = {"house_numbers": self.convert_house_numbers(int(goal.house_number.iloc[0])),
                        "street_names": self.convert_street_name(goal.street_name.iloc[0])}

        print('GOAL: ' + str(goal.house_number.iloc[0]) + ', ' + str(goal.street_name.iloc[0]) + ' street')
        return goal_idx, goal_address, goal_dir

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'LEFT_BIG': ord('a'),
            'LEFT_SMALL': ord('q'),
            'FORWARD': ord('w'),
            'RIGHT_SMALL': ord('e'),
            'RIGHT_BIG': ord('d'),
        }
        KEYWORD_TO_KEY['NOOP'] = ord('n')
        KEYWORD_TO_KEY['DONE'] = ord('s')

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action

    def get_visible_text(self, x, w):
        visible_text = {}
        house_numbers = []
        street_signs = []
        subset = self.meta_df.loc[self.agent_loc, ["house_number", "street_name", "obj_type", "x_min", "x_max"]]
        for idx, row in subset.iterrows():
            if x < row.x_min and x + w > row.x_max:
                if row.obj_type == "house_number":
                    house_numbers.append(row.house_number)
                elif row.obj_type == "street_sign":
                    street_signs.append(row.street_name)

        visible_text["house_numbers"] = house_numbers
        visible_text["street_names"] = street_signs
        return visible_text