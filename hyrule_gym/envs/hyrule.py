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

class HyruleEnv(gym.GoalEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    class Actions(enum.IntEnum):
        LEFT_BIG = 0
        LEFT_SMALL = 1
        FORWARD = 2
        RIGHT_SMALL = 3
        RIGHT_BIG = 4

    class NoopActions(enum.IntEnum):
        LEFT_BIG = 0
        LEFT_SMALL = 1
        FORWARD = 2
        RIGHT_SMALL = 3
        RIGHT_BIG = 4
        NOOP = 6

    class DoneActions(enum.IntEnum):
        LEFT_BIG = 0
        LEFT_SMALL = 1
        FORWARD = 2
        RIGHT_SMALL = 3
        RIGHT_BIG = 4
        DONE = 5

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
        res = self.meta_df[self.meta_df.obj_type == 'street_sign'].street_name.unique()
        res = (res == street_name).astype(int)
        return res

    def __init__(self, path="/mini-corl/processed/", obs_type='image', obs_shape=(4, 84, 84),
                 shaped_reward=True, can_noop=False, can_done=False, store_test=False, test_mode=False):
        self.viewer = None
        self.needs_reset = True
        self.can_noop = can_noop
        self.can_done = can_done
        if can_noop:
            HyruleEnv.Actions = HyruleEnv.NoopActions
        elif can_done:
            HyruleEnv.Actions = HyruleEnv.DoneActions

        self._action_set = HyruleEnv.Actions
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.float32) # spaces.dict goes here
        import pdb; pdb.set_trace()
        path = _ROOT + path
        f = gzip.GzipFile(path + "images.pkl.gz", "r")
        self.images_df = pickle.load(f)
        f.close()
        self.meta_df = pd.read_hdf(path + "meta.hdf5", key='df', mode='r')
        self.G = nx.read_gpickle(path + "graph.pkl")

        self.num_streets = self.meta_df[self.meta_df.obj_type == 'street_sign'].street_name.unique().size
        self.curriculum_learning = False
        self.agent_loc = np.random.choice(self.meta_df.frame)
        self.agent_dir = 0
        self.difficulty = 0
        self.weighted = True

        self.shaped_reward = shaped_reward
        self.max_num_steps = 100
        self.num_steps_taken = 0

        self.store_test = store_test
        self.test_mode = test_mode

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

    def select_goal(self, same_segment=True, one_segment=True, difficulty=0):
        goals = self.meta_df[self.meta_df.is_goal == True]
        G = self.G.copy()
        if same_segment:
            if one_segment:
                frames = self.meta_df[(self.meta_df.type == "street_segment") & self.meta_df.frame.isin(goals.frame) & (self.meta_df.group == 1)].frame
                self.max_num_steps = self.meta_df[(self.meta_df.type == "street_segment") & (self.meta_df.group == 1)].shape[0]
            else:
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
        # randomly selects a node n-transitions from the goal node
        # if int(difficulty/4) == 0:
        #     nodes = [goal_idx]
        # if int(difficulty/4) >= 1:
        #     nodes = set(nx.ego_graph(G, goal_idx, radius=int(difficulty/4)))
        #     nodes -= set(nx.ego_graph(G, goal_idx, radius=int(difficulty-1 / 4)))
        # if self.curriculum_learning:
        #     self.agent_loc = np.random.choice(list(nodes)) if len(list(nodes)) > 0 else np.random.choice(list(G.nodes))
        #     if difficulty == 0:
        #         self.agent_dir = int(goal_dir/22.5)*22.5
        #     elif difficulty <=3:
        #         self.agent_dir = (int(goal_dir/22.5)-np.random.choice(range(-difficulty, difficulty)))*22.5
        #     else:
        #         self.agent_dir = 22.5 * np.random.choice(range(-8, 8))
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

    def set_difficulty(self, difficulty, weighted=False):
        self.difficulty = difficulty

    def step(self, a):
        done = False
        reward = 0.0
        action = self._action_set(a)
        oracle = False
        if oracle:
            shortest_path = self.shortest_path_length()
            action = shortest_path[0]
        image, x, w = self._get_image()
        visible_text = self.get_visible_text(x, w)

        if not self.can_done and self.is_successful_trajectory(x):
            done = True
            reward = self.compute_reward(x, {}, done)
        elif action == self.Actions.FORWARD:
            self.transition()
        elif self.can_done and action == self.Actions.DONE:
            done = True
            reward = self.compute_reward(x, {}, done)
            # print("Mission reward: " + str(reward))
        else:
            self.turn(action)

        if self.shaped_reward:
            reward = self.compute_reward(x, {}, done)
            #print("Current reward: " + str(reward))
        self.agent_gps = self.sample_gps(self.meta_df.loc[self.agent_loc])
        rel_gps = [self.target_gps[0] - self.agent_gps[0], self.target_gps[1] - self.agent_gps[1],
                   self.target_gps[0], self.target_gps[1]]
        obs = {"image": image, "mission": self.goal_address, "rel_gps": rel_gps, "visible_text": visible_text}
        self.num_steps_taken += 1
        if self.num_steps_taken >= self.max_num_steps and done == False:
            done = True
            reward = 0.0
        s = "obs: "
        for k, v in obs.items():
            if k != "image":
                s = s + ", " + str(k) + ": " + str(v)
        if done:
            self.needs_reset = True
        # print(self.agent_loc)
        # print(s)
        # print(obs)
        obs = self.obs_wrap(obs)
        return obs, reward, done, {}


    def _get_image(self, high_res=False, plot=False):
        if high_res:
            path = "data/data/mini-corl/panos/pano_"+ str(int(30*self.G.nodes[self.agent_loc]['timestamp'])).zfill(6) + ".png"
            img = cv2.resize(cv2.imread(path)[:, :, ::-1], (960, 480))
            obs_shape = (480, 480, 3)
        else:
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

        if plot:
            _, ax = plt.subplots(figsize=(18, 18))
            ax.imshow(res_img.astype(int))
            plt.show()
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
        if self.test_mode:
            self.load_test_task()
        else:
            self.goal_idx, self.goal_address, self.goal_dir = self.select_goal(same_segment=True, difficulty=self.difficulty)
            self.prev_spl = len(self.shortest_path_length())
            if self.store_test:
                self.store_test_task()
                self.store_test = False
        self.start_spl = self.prev_spl
        self.agent_gps = self.sample_gps(self.meta_df.loc[self.agent_loc])
        self.target_gps = self.sample_gps(self.meta_df.loc[self.goal_idx], scale=3.0)
        image, x, w = self._get_image()
        rel_gps = [self.target_gps[0] - self.agent_gps[0], self.target_gps[1] - self.agent_gps[1],
                   self.target_gps[0], self.target_gps[1]]
        obs = {"image": image, "mission": self.goal_address, "rel_gps": rel_gps, "visible_text": self.get_visible_text(x, w)}
        obs = self.obs_wrap(obs)
        return obs

    def obs_wrap(self, obs):
        coord_holder = np.zeros((1, 84, 84), dtype=np.float32)
        coord_holder[0, 0, :4] = obs['rel_gps']
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
                if self.can_done:
                    actions.append(self.Actions.DONE)
        # print("SPL:", len(actions))
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
        if self.shaped_reward:
            cur_spl = len(self.shortest_path_length())
            # print("SPL:", cur_spl)
            if done and self.is_successful_trajectory(x):
                reward = 2.0
            elif done and not self.is_successful_trajectory(x):
                reward = -2.0
            elif self.prev_spl - cur_spl > 0:
                reward = 1 #- (self.num_steps_taken / self.start_spl)
            elif self.prev_spl - cur_spl <= 0:
                reward = -1
            else:
                reward = 0.0
            self.prev_spl = cur_spl
            # print("reward: " + str(reward))
            return reward
        if self.is_successful_trajectory(x):
            return 1.0
        return 0.0

    def is_successful_trajectory(self, x):
        subset = self.meta_df.loc[self.agent_loc, ["frame", "obj_type", "house_number", "x_min", "x_max"]]
        label = subset[(subset.house_number == self.goal_id.iloc[0]) & (subset.obj_type == "door")]#) & (subset.obj_type == "door")]].any()
        x_min = label.x_min.get(0, 0) if type(label.x_min) == pd.Series else label.x_min
        x_max = label.x_max.get(0, 0) if type(label.x_max) == pd.Series else label.x_max
        if label.any().any() and x < x_min and x + 84 > x_max:
            #print("achieved goal")
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
        if self.can_noop:
            KEYWORD_TO_KEY['NOOP'] = ord('n')
        if self.can_done:
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

    def store_test_task(self):
        path = "/data/data/mini-corl/processed/"
        agent_info = {"idx": self.agent_loc, "dir": self.agent_dir}
        goal_info = {"idx": self.goal_idx, "dir": self.goal_dir, "address": self.goal_address, "goal_id": self.goal_id}
        test_task = {"agent_info": agent_info, "goal_info": goal_info, "spl": self.prev_spl}
        np.save(self.mydir+path + "test_task.npy", test_task)

    def load_test_task(self):
        test_task = np.load(self.mydir+"/data/data/mini-corl/processed/test_task.npy")
        self.agent_loc = test_task.item().get('agent_info')['idx']
        self.agent_dir = test_task.item().get('agent_info')['dir']
        self.goal_idx = test_task.item().get('goal_info')['idx']
        self.goal_dir = test_task.item().get('goal_info')['dir']
        self.goal_address = test_task.item().get('goal_info')['address']
        self.goal_id = test_task.item().get('goal_info')['goal_id']
        self.prev_spl = test_task.item().get('spl')
        #print("Test Task SPL:", test_task.item().get('spl'))
