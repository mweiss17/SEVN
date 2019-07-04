""" This is the simulator for NAVI project. It defines the action and observation spaces, tracks the agent's state, and specifies game logic. """
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

ACTION_MEANING = {
    0: 'LEFT_BIG',
    1: 'LEFT_SMALL',
    2: 'FORWARD',
    3: 'RIGHT_SMALL',
    4: 'RIGHT_BIG',
    5: 'DONE',
    6: 'NOOP',
}

class HyruleEnvShaped(gym.GoalEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    class Actions(enum.IntEnum):
        LEFT_BIG = 0
        LEFT_SMALL = 1
        FORWARD = 2
        RIGHT_SMALL = 3
        RIGHT_BIG = 4

    @classmethod
    def norm_angle(cls, x):
        # Utility function to keep some angles in the space of -180 to 180 degrees
        if x > 180:
            x = -360 + x
        elif x < -180:
            x = 360 + x
        return x

    @classmethod
    def convert_house_numbers(cls, num):
        res = np.zeros((4, 10))
        for col, row in enumerate(str(num)):
            res[col, int(row)] = 1
        return res.reshape(-1)

    def convert_street_name(self, street_name):
        assert street_name in self.all_street_names
        return (self.all_street_names == street_name).astype(int)

    def __init__(self, path="/corl/processed/", obs_shape=(4, 84, 84), use_image_obs=False, use_gps_obs=False, use_visible_text_obs=False):
        self.viewer = None
        self.use_image_obs = use_image_obs
        self.use_gps_obs = use_gps_obs
        self.use_visible_text_obs = use_visible_text_obs
        self.needs_reset = True
        self._action_set = HyruleEnvShaped.Actions
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.float32) # spaces.dict goes here
        path = _ROOT + path
        f = gzip.GzipFile(path + "images.pkl.gz", "r")
        self.images_df = pickle.load(f)
        f.close()
        self.meta_df = pd.read_hdf(path + "meta.hdf5", key='df', mode='r')
        self.G = nx.read_gpickle(path + "graph.pkl")

        self.all_street_names = self.meta_df[self.meta_df.is_goal == True].street_name.unique()
        self.num_streets = self.all_street_names.size
        self.agent_loc = np.random.choice(self.meta_df.frame)
        self.agent_dir = 0

        self.max_num_steps = 100
        self.num_steps_taken = 0

    def turn(self, action):
        action = self._action_set(action)
        if action == self.Actions.LEFT_BIG:
            self.agent_dir += 67.5
        if action == self.Actions.LEFT_SMALL:
            self.agent_dir += 22.5
        if action == self.Actions.RIGHT_SMALL:
            self.agent_dir -= 22.5
        if action == self.Actions.RIGHT_BIG:
            self.agent_dir -= 67.5
        self.agent_dir = self.norm_angle(self.agent_dir)


    def get_angle_between_nodes(self, n1, n2, use_agent_dir=True):
        x = self.G.nodes[n1]['coords'][0] - self.G.nodes[n2]['coords'][0]
        y = self.G.nodes[n1]['coords'][1] - self.G.nodes[n2]['coords'][1]
        angle = (math.atan2(y, x) * 180 / np.pi) + 180
        if use_agent_dir:
            return np.abs(self.norm_angle(angle - self.agent_dir))
        else:
            return angle

    def select_goal(self, same_segment=True):
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
        self.agent_loc = np.random.choice(segment_panos.frame.unique())
        goal_address = {"house_numbers": self.convert_house_numbers(int(goal.house_number.iloc[0])),
                        "street_names": self.convert_street_name(goal.street_name.iloc[0])}
        return goal_idx, goal_address, goal_dir


    def transition(self):
        """
        This function calculates the angles to the other panos
        then transitions to the one that is closest to the agent's current direction
        """
        neighbors = {}
        for n in [edge[1] for edge in list(self.G.edges(self.agent_loc))]:
            neighbors[n] = self.get_angle_between_nodes(n, self.agent_loc)

        if neighbors[min(neighbors, key=neighbors.get)] > 45:
            return

        self.agent_loc = min(neighbors, key=neighbors.get)

    def step(self, a):
        done = False
        reward = 0.0
        self.num_steps_taken += 1
        action = self._action_set(a)
        oracle = False
        if oracle:
            shortest_path = self.shortest_path_length()
            action = shortest_path[0]
        image, x, w = self._get_image()
        visible_text = self.get_visible_text(x, w)
        was_successful_trajectory = False

        if self.is_successful_trajectory(x):
            done = True
            was_successful_trajectory = True
        elif self.num_steps_taken >= self.max_num_steps and done == False:
            done = True
        elif action == self.Actions.FORWARD:
            self.transition()
        else:
            self.turn(action)
        reward = self.compute_reward(x, {}, done)

        self.agent_gps = self.sample_gps(self.meta_df.loc[self.agent_loc])
        rel_gps = [self.target_gps[0] - self.agent_gps[0], self.target_gps[1] - self.agent_gps[1]]
        obs = {"image": image, "mission": self.goal_address, "rel_gps": rel_gps, "visible_text": visible_text}
        obs = self.obs_wrap(obs)

        info = {}
        if done:
            info["was_successful_trajectory"] = was_successful_trajectory
            self.needs_reset = True

        return obs, reward, done, info


    def _get_image(self, high_res=False, plot=False):
        img = self.images_df[self.meta_df.loc[self.agent_loc, 'frame'][0]]
        obs_shape = self.observation_space.shape

        pano_rotation = self.norm_angle(self.meta_df.loc[self.agent_loc, 'angle'][0] + 90)
        w = obs_shape[1]
        y = img.shape[0] - obs_shape[1]
        h = obs_shape[2]
        x = int((self.norm_angle(-self.agent_dir + pano_rotation) + 180)/360 * img.shape[1])
        img = img.transpose()
        if (x + w) % img.shape[1] != (x + w):
            res_img = np.zeros((3, 84, 84))
            offset = img.shape[1] - (x % img.shape[1])
            res_img[:, :offset] = img[y:y+h, x:x + offset]
            res_img[:, offset:] = img[y:y+h, :(x + w) % img.shape[1]]
        else:
            res_img = img[:, x:x + w]

        return res_img, x, w

    def get_visible_text(self, x, w):
        visible_text = {}
        house_numbers = []
        street_signs = []
        subset = self.meta_df.loc[self.agent_loc, ["house_number", "street_name", "obj_type", "x_min", "x_max"]]
        for idx, row in subset.iterrows():
            if x < row.x_min and x + w > row.x_max:
                if row.obj_type == "house_number":
                    house_numbers.append(self.convert_house_numbers(row.house_number))
                elif row.obj_type == "street_sign":
                    street_signs.append(self.convert_street_name(row.street_name))

        temp = np.zeros(120)
        if len(house_numbers) != 0:
            nums = np.hstack(house_numbers)[:120]
            temp[:nums.size] = nums
        visible_text["house_numbers"] = temp

        temp = np.zeros(2 * self.num_streets)
        if len(street_signs) != 0:
            nums = np.hstack(street_signs)[:6]
            temp[:nums.size] = nums
        visible_text["street_names"] = temp
        return visible_text

    def sample_gps(self, groundtruth, scale=1):
        coords = groundtruth[['x', 'y']]
        gps_scale = 100.0  # TODO: Arbitrary. Need a better normalizing value here. Requires min-max from dataframe.
        x = (coords.at[0, 'x'] + np.random.normal(loc=0.0, scale=scale)) / gps_scale
        y = (coords.at[0, 'y'] + np.random.normal(loc=0.0, scale=scale)) / gps_scale
        return (x, y)

    def reset(self):
        self.needs_reset = False
        self.num_steps_taken = 0
        self.goal_idx, self.goal_address, self.goal_dir = self.select_goal(same_segment=True)
        self.prev_spl = len(self.shortest_path_length())
        self.start_spl = self.prev_spl
        self.agent_gps = self.sample_gps(self.meta_df.loc[self.agent_loc])
        self.target_gps = self.sample_gps(self.meta_df.loc[self.goal_idx], scale=3.0)
        image, x, w = self._get_image()
        rel_gps = [self.target_gps[0] - self.agent_gps[0], self.target_gps[1] - self.agent_gps[1]]
        obs = {"image": image, "mission": self.goal_address, "rel_gps": rel_gps, "visible_text": self.get_visible_text(x, w)}
        obs = self.obs_wrap(obs)
        return obs

    def obs_wrap(self, obs):
        coord_holder = np.zeros((1, 84, 84), dtype=np.float32)

        if self.use_gps_obs:
            coord_holder[0, 0, :2] = obs['rel_gps']

        coord_holder[0, 0, 3] = self.num_streets
        if self.use_visible_text_obs:
            coord_holder[0, 1, :2 * self.num_streets] = obs['visible_text']['street_names']
            coord_holder[0, 2, :] = obs['visible_text']['house_numbers'][:84]
            coord_holder[0, 3, :36] = obs['visible_text']['house_numbers'][84:120]
        if not self.use_image_obs:
            obs['image'] = np.zeros((3, 84, 84))

        coord_holder[0, 4, :40] = obs['mission']['house_numbers']
        coord_holder[0, 4, 40:40 + self.num_streets] = obs['mission']["street_names"]

        out = np.concatenate((obs['image'], coord_holder), axis=0)
        return out

    def angles_to_turn(self, cur, target):
        go_left = []
        go_right = []
        temp = cur
        while np.abs(target - cur) > 67.5:
            cur = (cur - 67.5) % 360
            go_right.append(self.Actions.RIGHT_BIG)

        while np.abs(target - cur) > 22.5:
            cur = (cur - 22.5) % 360
            go_right.append(self.Actions.RIGHT_SMALL)
        cur = temp

        while np.abs(target - cur) > 67.5:
            cur = (cur + 67.5) % 360
            go_left.append(self.Actions.LEFT_BIG)

        while np.abs(target - cur) > 22.5:
            cur = (cur + 22.5) % 360
            go_left.append(self.Actions.LEFT_SMALL)

        if len(go_left) > len(go_right):
            return go_right
        return go_left

    def shortest_path_length(self):
        # finds a minimal trajectory to navigate to the target pose
        # target_index = self.coords_df[self.coords_df.frame == int(target_node_info['timestamp'] * 30)].index.values[0]
        cur_node = self.agent_loc
        cur_dir = self.agent_dir + 180
        target_node = self.goal_idx
        path = nx.shortest_path(self.G, cur_node, target=target_node)
        actions = []
        for idx, node in enumerate(path):
            if idx + 1 != len(path):
                target_dir = self.get_angle_between_nodes(node, path[idx + 1], use_agent_dir=False)
                actions.extend(self.angles_to_turn(cur_dir, target_dir))
                cur_dir = target_dir
                actions.append(self.Actions.FORWARD)
            else:
                actions.extend(self.angles_to_turn(cur_dir, self.goal_dir + 180))
        return actions


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

    def is_successful_trajectory(self, x):
        subset = self.meta_df.loc[self.agent_loc, ["frame", "obj_type", "house_number", "x_min", "x_max"]]
        label = subset[(subset.house_number == self.goal_id.iloc[0]) & (subset.obj_type == "door")]
        x_min = label.x_min.get(0, 0) if type(label.x_min) == pd.Series else label.x_min
        x_max = label.x_max.get(0, 0) if type(label.x_max) == pd.Series else label.x_max
        if label.any().any() and x < x_min and x + 84 > x_max:
            return True
        return False

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

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'LEFT_BIG': ord('a'),
            'LEFT_SMALL': ord('q'),
            'FORWARD': ord('w'),
            'RIGHT_SMALL': ord('e'),
            'RIGHT_BIG': ord('d'),
        }

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
