from __future__ import print_function, division
import os
import math
import h5py
import zipfile
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import gym
from SEVN_gym.envs.utils import Actions, ACTION_MEANING, get_data_path, HighResDataset, norm_angle_360
from gym import spaces
from matplotlib.collections import LineCollection
from SEVN_gym.envs import utils, wrappers
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class SEVNActionspace(gym.GoalEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 obs_shape=(8, 84, 224),
                 use_image_obs=True,
                 use_gps_obs=True,
                 use_visible_text_obs=True,
                 split="Test",
                 reward_type=None,
                 continuous=False,
                 concurrent_access=False,
                 high_res=False,
                 curriculum=True,
                 curriculum_level=2):

        # in order to speed up training/NFS file access, you can make 5 copies of
        # SEVN_gym/data0 and call them SEVN_gym/data0 to ...data4. Then this script will
        # randomly pick one of those and reduce concurrent access
        if concurrent_access:
            DATA_PATH = get_data_path()
        else:
            from SEVN_gym.envs.utils import DATA_PATH

        print(f'Booting environment from {DATA_PATH} with shaped reward,' +
              f' image_obs: {use_image_obs}, gps: {use_gps_obs},' +
              f' visible_text: {use_visible_text_obs}')

        # Initialize environment
        self.viewer = None
        self.high_res = high_res
        if high_res:
            obs_shape = (8, 1280, 1280)
        self.continuous = continuous
        self.use_image_obs = use_image_obs
        self.use_gps_obs = use_gps_obs
        self.use_visible_text_obs = use_visible_text_obs
        self.reward_type = reward_type
        self.goal_type = "segment"
        self.curriculum = curriculum
        self.curriculum_level = curriculum_level
        self.needs_reset = True
        self.plot_ready = False
        self.is_explorer = False
        self.action_space = spaces.Discrete(12)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.float32)
        self.agent_dir = 0
        self.num_steps_taken = 0
        self.prev_spl = 100000
        self.goal_hn = -1

        # Load data
        if not os.path.isfile(os.path.join(DATA_PATH, 'images.hdf5')) \
                or not os.path.isfile(os.path.join(DATA_PATH, 'graph.pkl')) \
                or not os.path.isfile(os.path.join(DATA_PATH, 'label.hdf5')) \
                or not os.path.isfile(os.path.join(DATA_PATH, 'coord.hdf5')):
            zipfile.ZipFile(os.path.join(DATA_PATH,
                                         'dataset.zip')).extractall(DATA_PATH)

        f = h5py.File(os.path.join(DATA_PATH, 'images.hdf5'), 'r')
        self.images = f["images"]
        self.frame_key = {int(k): i for i, k in enumerate(f['frames'][:])}
        if high_res:
            self.images = HighResDataset(self.frame_key)

        self.label_df = pd.read_hdf(
            os.path.join(DATA_PATH, 'label.hdf5'), key='df', mode='r')
        self.coord_df = pd.read_hdf(
            os.path.join(DATA_PATH, 'coord.hdf5'), key='df', mode='r')
        self.G = nx.read_gpickle(os.path.join(DATA_PATH, 'graph.pkl'))

        if split == 'Test':
            self.bad_indices = set(self.coord_df.index).difference(
                set(utils.filter_for_test(self.coord_df).index))

        if split == 'Train':
            self.bad_indices = utils.filter_for_test(self.coord_df).index

        if split == 'trainv2':
            self.bad_indices = utils.filter_for_test(self.coord_df).index
            self.bad_indices = set(self.bad_indices).union(
                set(utils.filter_for_trainv2(self.coord_df).index))

        # Set data0-dependent variables
        self.max_num_steps = self.curriculum_level * 2 #self.coord_df[self.coord_df.type == 'street_segment'].groupby('group').count().max().iloc[0]
        self.all_street_names = self.label_df.street_name.dropna().unique()
        self.num_streets = self.all_street_names.size
        self.x_scale = self.coord_df.x.max() - self.coord_df.x.min()
        self.y_scale = self.coord_df.y.max() - self.coord_df.y.min()
        self.agent_loc = np.random.choice(self.coord_df.index)

    def graph_plotting(self, figname):
        plt.ion()
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        self.pos = {k: v.get('coords')[0:2] for k, v in self.G.nodes(data=True)}
        # nx.draw(self.G, pos, node_color='r', node_size=1)
        nodelist = list(self.G)
        self.xy = np.asarray([self.pos[v] for v in nodelist])
        self.corners = np.asarray([
            self.pos[node]
            for node in list(self.G)
            if node in self.coord_df[self.coord_df.type == 'intersection'].index
        ])
        self.streets = np.asarray([
            self.pos[node] for node in list(self.G) if node in self.coord_df[
                self.coord_df.type == 'street_segment'].index
        ])
        edgelist = list(self.G.edges())
        edge_pos = np.asarray([
            (self.pos[e[0]], self.pos[e[1]]) for e in edgelist
        ])
        self.edge_collection = LineCollection(edge_pos)
        self.edge_collection.set_zorder(1)  # edges go behind nodes
        self.ax.scatter(self.corners[:, 0], self.corners[:, 1], c='#fde724')
        self.ax.scatter(self.streets[:, 0], self.streets[:, 1], c='#79d151')
        self.ax.add_collection(self.edge_collection)
        plt.savefig(figname)

    def select_goal(self):
        goals = self.label_df.loc[self.label_df['is_goal'] == True]
        frames = self.coord_df[
            (self.coord_df.type == 'street_segment')
            & self.coord_df.index.isin(goals.index)
            & ~self.coord_df.index.isin(self.bad_indices)].index
        goals_on_street_segment = goals[goals.index.isin(frames)]
        # goal = goals_on_street_segment.loc[goals_on_street_segment.index.values.tolist()[0]]
        goal = goals_on_street_segment.loc[np.random.choice(
            goals_on_street_segment.index.values.tolist())]
        if len(goal.shape) > 1:
            goal = goal.iloc[np.random.randint(len(goal))]
        self.goal_hn = goal.house_number
        candidates = utils.get_candidate_start_nodes(self.curriculum_level, goal.name, self.G)
        self.agent_loc = np.random.choice(candidates)
        goal_address = {
            'house_numbers':
                utils.convert_house_numbers(self.goal_hn),
            'street_names':
                utils.convert_street_name(goal.street_name,
                                          self.all_street_names)
        }

        return goal.name, goal_address


    def compute_reward(self, action, done, obs=None):
        self.prev_sp = nx.shortest_path(self.G, self.agent_loc, target=self.goal_idx)
        self.prev_spl = len(self.prev_sp) - 1
        if self.prev_spl == 0:
            return 1
        else:
            return -1 / self.max_num_steps

    def transition(self, action):
        '''
        This function calculates the angles to the other panos then
        transitions to the one that is closest to the agent's current direction
        '''
        neighbors = {}
        for n in [edge[1] for edge in list(self.G.edges(self.agent_loc))]:
            neighbors[n] = np.abs(norm_angle_360(utils.get_angle_between_nodes(self.G, self.agent_loc, n)) - action)

        self.agent_loc = min(neighbors, key=neighbors.get)

    def step(self, a):
        done = False
        # if a[0] == 12:
        #     done = True
        was_successful_trajectory = False
        oracle = False
        self.num_steps_taken += 1
        action = 360 * (a[0]) / 11 # Translate to unit circle angle
        if oracle:
            action = next(iter(self.shortest_path_length()), None)
        self.transition(action)

        # Get new observation
        image = self._get_image()
        visible_text = self._get_visible_text(0, 224)
        self.agent_gps = utils.sample_gps(self.coord_df.loc[self.agent_loc],
                                          self.x_scale, self.y_scale)
        rel_gps = [
            self.target_gps[0] - self.agent_gps[0],
            self.target_gps[1] - self.agent_gps[1], self.target_gps[0],
            self.target_gps[1]
        ]
        obs = {
            'image': image,
            'mission': self.goal_address,
            'rel_gps': rel_gps,
            'visible_text': visible_text
        }

        # Compute Reward
        reward = self.compute_reward(action, done, obs)
        obs = wrappers.wrap_obs_pano(obs, self.use_gps_obs,
                                self.use_visible_text_obs, self.use_image_obs,
                                True, self.num_streets)

        if self.is_successful_trajectory():
            done = True
            was_successful_trajectory = True
        elif self.num_steps_taken >= self.max_num_steps:
            done = True

        info = {}
        if done:
            info['was_successful_trajectory'] = was_successful_trajectory
            self.needs_reset = True
        return obs, reward, done, info

    def _get_image(self):
        img = self.images[self.frame_key[self.agent_loc]]
        return img

    def _get_visible_text(self, x, w):
        if self.agent_loc not in self.label_df.index:
            return {
                'street_names': np.zeros(2 * len(self.all_street_names)),
                'house_numbers': np.zeros(120)
            }
        subset = self.label_df.loc[self.agent_loc, [
            'house_number', 'street_name', 'obj_type', 'x_min', 'x_max'
        ]]
        house_numbers, street_names = utils.extract_text(
            x, w, subset, self.high_res)
        house_numbers = \
            [utils.convert_house_numbers(num) for num in house_numbers]
        street_names = \
            [utils.convert_street_name(name, self.all_street_names) for
             name in street_names]
        visible_text = utils.stack_text(house_numbers, street_names,
                                        self.num_streets)
        return visible_text

    def reset(self):
        self.needs_reset = False
        self.min_angle = np.inf
        self.min_spl = np.inf
        self.moved_forward = False
        self.num_steps_taken = 0
        self.goal_idx, self.goal_address = self.select_goal()
        self.prev_sp = nx.shortest_path(
            self.G, self.agent_loc, target=self.goal_idx)
        self.prev_spl = len(self.prev_sp) - 1
        self.agent_gps = utils.sample_gps(self.coord_df.loc[self.agent_loc],
                                          self.x_scale, self.y_scale)
        self.target_gps = utils.sample_gps(self.coord_df.loc[self.goal_idx],
                                           self.x_scale, self.y_scale)
        image = self._get_image()
        rel_gps = [
            self.target_gps[0] - self.agent_gps[0],
            self.target_gps[1] - self.agent_gps[1], self.target_gps[0],
            self.target_gps[1]
        ]
        obs = {
            'image': image,
            'mission': self.goal_address,
            'rel_gps': rel_gps,
            'visible_text': self._get_visible_text(0, 224)
        }
        obs = wrappers.wrap_obs_pano(obs, self.use_gps_obs,
                                self.use_visible_text_obs, self.use_image_obs,
                                True, self.num_streets)
        return obs

    def shortest_path_length(self):
        '''
        Finds a minimal trajectory to navigate to the target pose.
        '''
        path = nx.shortest_path(self.G, self.agent_loc, target=self.goal_idx)
        target_dir = utils.get_angle_between_nodes(self.G, self.agent_loc, path[1])
        return [norm_angle_360(target_dir)] # [int(target_dir /360 * 12)]

    def is_successful_trajectory(self):
        if not self.agent_loc == self.goal_idx or self.is_explorer:
            return False
        return True

    def render(self, mode='human', clear=False, first_time=False):
        img = self._get_image()

        if first_time or not self.plot_ready:
            plt.ion()
            self.fig, self.ax = plt.subplots(nrows=1, ncols=2)
            self.ax[0] = self.ax[0].imshow(
                np.zeros((84, 224, 3)),
                interpolation='none',
                animated=True,
                label='ladida')
            self.plot = plt.gca()
        if clear or not self.plot_ready:
            self.plot_ready = True
            self.plot.cla()
            self.pos = {
                k: v.get('coords')[0:2] for k, v in self.G.nodes(data=True)
            }
            # nx.draw(self.G, pos, node_color='r', node_size=1)
            nodelist = list(self.G)
            self.xy = np.asarray([self.pos[v] for v in nodelist])
            self.corners = np.asarray([
                self.pos[node] for node in list(self.G) if node in
                self.coord_df[self.coord_df.type == 'intersection'].index
            ])
            self.streets = np.asarray([
                self.pos[node] for node in list(self.G) if node in
                self.coord_df[self.coord_df.type == 'street_segment'].index
            ])
            edgelist = list(self.G.edges())
            edge_pos = np.asarray([
                (self.pos[e[0]], self.pos[e[1]]) for e in edgelist
            ])
            self.edge_collection = LineCollection(edge_pos)
            self.edge_collection.set_zorder(1)  # edges go behind nodes
            self.ax[1].scatter(
                self.corners[:, 0], self.corners[:, 1], c='#fde724')
            self.ax[1].scatter(
                self.streets[:, 0], self.streets[:, 1], c='#79d151')
            self.ax[1].add_collection(self.edge_collection)
        agent_loc = self.pos[self.agent_loc]
        goal_loc = self.pos[self.goal_idx]
        self.agent_point = self.ax[1].plot(
            agent_loc[0], agent_loc[1], color='b', marker='o')
        self.ax[1].plot(goal_loc[0], goal_loc[1], color='r', marker='o')
        self.ax[0].set_data(utils.denormalize_image(img))
        plt.pause(0.01)
        self.agent_point[0].remove()
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


    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            '1': ord('1'),
            '2': ord('2'),
            '3': ord('3'),
            '4': ord('4'),
            '5': ord('5'),
            '6': ord('6'),
            '7': ord('7'),
            '8': ord('8'),
            '9': ord('9'),
            '10': ord('-'),
            '11': ord('='),
            '12': ord('`'),
        }

        return KEYWORD_TO_KEY
