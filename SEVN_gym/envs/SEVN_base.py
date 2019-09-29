from __future__ import print_function, division
import enum
import os
import time
import math
import h5py
import zipfile
import numpy as np
import pandas as pd
import dask.array as da
import networkx as nx
import academictorrents as at
import matplotlib.pyplot as plt
import gym
from gym import spaces
from matplotlib.collections import LineCollection
from SEVN_gym.data import DATA_PATH
from SEVN_gym.envs import utils, wrappers


ACTION_MEANING = {
    0: 'LEFT_BIG',
    1: 'LEFT_SMALL',
    2: 'FORWARD',
    3: 'RIGHT_SMALL',
    4: 'RIGHT_BIG',
    5: 'DONE',
    6: 'NOOP',
    7: 'READ'
}


class SEVNBase(gym.GoalEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    class Actions(enum.IntEnum):
        LEFT_BIG = 0
        LEFT_SMALL = 1
        FORWARD = 2
        RIGHT_SMALL = 3
        RIGHT_BIG = 4
        DONE = 5
        NOOP = 6
        READ = 7

    def __init__(self, obs_shape=(4, 84, 84), use_image_obs=False,
                 use_gps_obs=False, use_visible_text_obs=False,
                 split="train", reward_type=None):

        print(f'Booting environment from {DATA_PATH} with shaped reward,' +
              f' image_obs: {use_image_obs}, gps: {use_gps_obs},' +
              f' visible_text: {use_visible_text_obs}')

        # Initialize environment
        self.viewer = None
        self.high_res = False
        self.use_image_obs = use_image_obs
        self.use_gps_obs = use_gps_obs
        self.use_visible_text_obs = use_visible_text_obs
        self.reward_type = reward_type
        self.needs_reset = True
        self._action_set = SEVNBase.Actions
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.float32)
        self.agent_dir = 0
        self.num_steps_taken = 0
        self.prev_spl = 100000
        self.goal_hn = -1
        self.SMALL_TURN_DEG = 22.5
        self.BIG_TURN_DEG = 67.5

        # Load data
        if not os.path.isfile(DATA_PATH + 'images.hdf5') \
            or not os.path.isfile(DATA_PATH + 'graph.pkl') \
            or not os.path.isfile(DATA_PATH + 'label.hdf5') \
            or not os.path.isfile(DATA_PATH + 'coord.hdf5'):
            zipfile.ZipFile(at.get("b9e719976cdedb94a25d2f162b899d5f0e711fe0", datastore=DATA_PATH)) \
                .extractall(DATA_PATH)
        f = h5py.File(DATA_PATH + 'images.hdf5', 'r')
        self.images = da.from_array(f["images"])
        self.frame_key = {int(k): i for i, k in enumerate(f['frames'][:])}
        self.label_df = pd.read_hdf(DATA_PATH + 'label.hdf5', key='df', mode='r')
        self.coord_df = pd.read_hdf(DATA_PATH + 'coord.hdf5', key='df', mode='r')
        self.G = nx.read_gpickle(DATA_PATH + 'graph.pkl')

        if split == 'Test':
            indices = self.coord_df.index
            self.coord_df = utils.filter_for_test(self.coord_df)
            to_remove = set(indices).difference(set(self.coord_df.index))
            self.label_df = self.label_df[self.label_df.index.isin(self.coord_df.index)]
            self.G.remove_nodes_from(to_remove)

        if split == 'Train':
            test_indices = utils.filter_for_test(self.coord_df).index
            self.coord_df = self.coord_df[~self.coord_df.index.isin(test_indices)]
            self.label_df = self.label_df[~self.label_df.index.isin(test_indices)]
            self.G.remove_nodes_from(test_indices)

        # Set data-dependent variables
        self.max_num_steps = \
            self.coord_df[self.coord_df.type == 'street_segment'].groupby('group').count().max().frame
        self.all_street_names = self.label_df.street_name.dropna().unique()
        self.num_streets = self.all_street_names.size
        self.x_scale = self.coord_df.x.max() - self.coord_df.x.min()
        self.y_scale = self.coord_df.y.max() - self.coord_df.y.min()
        self.agent_loc = np.random.choice(self.coord_df.frame)

    def turn(self, action):
        # Modify agent heading
        action = self._action_set(action)
        if action == self.Actions.LEFT_BIG:
            self.agent_dir += self.BIG_TURN_DEG
        if action == self.Actions.LEFT_SMALL:
            self.agent_dir += self.SMALL_TURN_DEG
        if action == self.Actions.RIGHT_SMALL:
            self.agent_dir -= self.SMALL_TURN_DEG
        if action == self.Actions.RIGHT_BIG:
            self.agent_dir -= self.BIG_TURN_DEG
        self.agent_dir = utils.norm_angle(self.agent_dir)

    def select_goal(self, same_segment=True):
        goals = self.label_df.loc[self.label_df['is_goal'] == True]
        if same_segment:
            frames = self.coord_df[(self.coord_df.type == 'street_segment') &
                                   self.coord_df.frame.isin(goals.frame)].frame
            goals_on_street_segment = goals[goals.frame.isin(frames)]
            goal = goals_on_street_segment.loc[np.random.choice(
                goals_on_street_segment.frame.values.tolist())]
            if len(goal.shape) > 1:
                goal = goal.iloc[np.random.randint(len(goal))]
            segment_group = self.coord_df[self.coord_df.frame == goal.frame].group.iloc[0]
            segment_panos = \
                self.coord_df[(self.coord_df.group == segment_group) &
                             (self.coord_df.type == 'street_segment')]
        else:
            goal = goals.loc[np.random.choice(goals.frame.values.tolist())]
        self.goal_hn = goal.house_number
        pano_rotation = utils.norm_angle(self.coord_df.loc[goal.frame].angle)
        label = self.label_df.loc[goal.frame]
        if isinstance(label, pd.DataFrame):
            label = label[label.is_goal].iloc[np.random.choice(label[label.is_goal].shape[0])]
        label_dir = (224-(label.x_min+label.x_max)/2) * 360/224-180
        goal_dir = utils.norm_angle(label_dir + pano_rotation)
        self.agent_dir = self.SMALL_TURN_DEG * np.random.choice(range(-8, 8))
        self.agent_loc = np.random.choice(segment_panos.frame.unique())
        goal_address = {'house_numbers': utils.convert_house_numbers(self.goal_hn),
                        'street_names': utils.convert_street_name(
                            goal.street_name, self.all_street_names)}
        return goal.frame, goal_address, goal_dir

    def transition(self):
        '''
        This function calculates the angles to the other panos then
        transitions to the one that is closest to the agent's current direction
        '''
        neighbors = {}
        for n in [edge[1] for edge in list(self.G.edges(self.agent_loc))]:
            neighbors[n] = np.abs(utils.norm_angle(
                utils.get_angle_between_nodes(self.G, self.agent_loc, n) -
                self.agent_dir))

        if neighbors[min(neighbors, key=neighbors.get)] > 45:
            return

        self.agent_loc = min(neighbors, key=neighbors.get)

    def step(self, a):
        done = False
        was_successful_trajectory = False
        oracle = False

        reward = 0.0
        self.num_steps_taken += 1
        action = self._action_set(a)
        if oracle:
            action = next(iter(self.shortest_path_length()), None)
        image, x, w = self._get_image()
        visible_text = self._get_visible_text(x, w)
        try:
            if self.is_successful_trajectory(x):
                done = True
                was_successful_trajectory = True
            elif self.num_steps_taken >= self.max_num_steps and done is False:
                done = True
            elif action == self.Actions.FORWARD:
                self.transition()
            else:
                self.turn(action)
        except Exception:
            self.is_successful_trajectory(x)

        reward = self.compute_reward(x, {}, done)

        self.agent_gps = utils.sample_gps(self.coord_df.loc[self.agent_loc],
                                          self.x_scale, self.y_scale)
        rel_gps = [self.target_gps[0] - self.agent_gps[0],
                   self.target_gps[1] - self.agent_gps[1],
                   self.target_gps[0], self.target_gps[1]]
        obs = {'image': image, 'mission': self.goal_address,
               'rel_gps': rel_gps, 'visible_text': visible_text}
        obs = wrappers.wrap_obs(obs, self.use_gps_obs,
                                self.use_visible_text_obs,
                                self.use_image_obs, True, self.num_streets)

        info = {}
        if done:
            info['was_successful_trajectory'] = was_successful_trajectory
            self.needs_reset = True

        return obs, reward, done, info

    def _get_image(self):
        img = self.images[self.frame_key[self.agent_loc]]
        obs_shape = self.observation_space.shape

        pano_rotation = self.coord_df.loc[self.agent_loc, 'angle']
        agent_dir = utils.norm_angle_360(self.agent_dir)
        normed_ang = ((agent_dir - pano_rotation) % 360)/360
        w = obs_shape[1]
        y = img.shape[0] - obs_shape[1]
        h = obs_shape[2]
        x = int(img.shape[1] - (
            (normed_ang*img.shape[1]) +
            img.shape[1]/2+obs_shape[1]/2) % img.shape[1])
        img = img.transpose()
        if (x + w) % img.shape[1] != (x + w):
            res_img = np.zeros((3, 84, 84))
            offset = img.shape[1] - (x % img.shape[1])
            res_img[:, :offset] = img[y:y+h, x:x + offset]
            res_img[:, offset:] = img[y:y+h, :(x + w) % img.shape[1]]
        else:
            res_img = img[:, x:x + w]

        return res_img, x, w

    def _get_visible_text(self, x, w):
        if self.agent_loc not in self.label_df.index:
            return {'street_names': np.zeros(2 * len(self.all_street_names)),
                    'house_numbers': np.zeros(120)}
        subset = self.label_df.loc[self.agent_loc, [
            'house_number', 'street_name', 'obj_type', 'x_min', 'x_max']]
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
        self.num_steps_taken = 0
        self.goal_idx, self.goal_address, self.goal_dir = \
            self.select_goal(same_segment=True)
        self.prev_spl = len(self.shortest_path_length())
        self.start_spl = self.prev_spl
        self.agent_gps = utils.sample_gps(self.coord_df.loc[self.agent_loc],
                                          self.x_scale, self.y_scale)
        self.target_gps = utils.sample_gps(self.coord_df.loc[self.goal_idx],
                                           self.x_scale, self.y_scale)
        image, x, w = self._get_image()
        rel_gps = [self.target_gps[0] - self.agent_gps[0],
                   self.target_gps[1] - self.agent_gps[1],
                   self.target_gps[0], self.target_gps[1]]
        obs = {'image': image,
               'mission': self.goal_address,
               'rel_gps': rel_gps,
               'visible_text': self._get_visible_text(x, w)}
        obs = wrappers.wrap_obs(obs, self.use_gps_obs,
                                self.use_visible_text_obs,
                                self.use_image_obs, True, self.num_streets)
        return obs

    def angles_to_turn(self, cur, target):
        angle = utils.smallest_angle(cur, target)
        turns = []
        if (np.sign(angle)) == 1:
            big_turns = int(angle / self.BIG_TURN_DEG)
            turns.extend([self.Actions.LEFT_BIG for i in range(big_turns)])
            cur = utils.norm_angle(cur + big_turns * self.BIG_TURN_DEG)
            small_turns = int((angle - big_turns * self.BIG_TURN_DEG) / self.SMALL_TURN_DEG)
            turns.extend([self.Actions.LEFT_SMALL for i in range(small_turns)])
            cur = utils.norm_angle(cur + small_turns * self.SMALL_TURN_DEG)
        else:
            angle = np.abs(angle)
            big_turns = int(angle / self.BIG_TURN_DEG)
            turns.extend([self.Actions.RIGHT_BIG for i in range(big_turns)])
            cur = utils.norm_angle(cur - big_turns * self.BIG_TURN_DEG)
            small_turns = int((angle - big_turns * self.BIG_TURN_DEG) / self.SMALL_TURN_DEG)
            turns.extend([
                self.Actions.RIGHT_SMALL for i in range(small_turns)])
            cur = utils.norm_angle(cur - small_turns * self.SMALL_TURN_DEG)
        return turns, cur

    def shortest_path_length(self):
        '''
        Finds a minimal trajectory to navigate to the target pose.
        '''
        cur_node = self.agent_loc
        cur_dir = self.agent_dir
        target_node = self.goal_idx
        path = nx.shortest_path(self.G, cur_node, target=target_node)
        actions = []
        for idx, node in enumerate(path):
            if idx + 1 != len(path):
                target_dir = utils.angle_to_node(self.G, node, path[idx + 1], self.SMALL_TURN_DEG)
                new_action, final_dir = self.angles_to_turn(cur_dir,
                                                            target_dir)
                actions.extend(new_action)
                cur_dir = final_dir
                actions.append(self.Actions.FORWARD)
            else:
                new_action, final_dir = self.angles_to_turn(cur_dir,
                                                            self.goal_dir)
                actions.extend(new_action)
        return actions

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

        if self.reward_type == 'Sparse' and reward != 2.0 and reward != -2.0:
            reward = 0
        self.prev_spl = cur_spl
        return reward

    def is_successful_trajectory(self, x):
        try:
            label = self.label_df.loc[self.agent_loc,
                                      ['frame', 'obj_type', 'house_number',
                                       'x_min', 'x_max']]
            if isinstance(label, pd.DataFrame):
                label = label[(label.house_number == self.goal_hn) &
                               (label.obj_type == 'door')]
            label = label.iloc[np.random.randint(len(label))]
            if label.empty:
                return False
            if x < label.x_min and x + 84 > label.x_max:
                return x < label.x_min and x + 84 > label.x_max
        except (KeyError, IndexError, AttributeError, ValueError) as e:
            return False

    def render(self, mode='human', clear=False, first_time=False):
        img, x, w = self._get_image()
        if first_time:
            plt.ion()
            self.fig, self.ax = plt.subplots(nrows=1, ncols=2)
            self.ax[0] = self.ax[0].imshow(
                np.zeros((84, 84, 3)),
                interpolation='none',
                animated=True,
                label='ladida')
            self.plot = plt.gca()
        if clear:
            self.plot.cla()
            self.pos = {k: v.get('coords')[0:2] for
                        k, v in self.G.nodes(data=True)}
            # nx.draw(self.G, pos, node_color='r', node_size=1)
            nodelist = list(self.G)
            self.xy = np.asarray([self.pos[v] for v in nodelist])
            self.corners = np.asarray([
                self.pos[node] for node in list(self.G) if node in
                self.coord_df[self.coord_df.type == 'intersection'].frame])
            self.streets = np.asarray([
                self.pos[node] for node in list(self.G) if node in
                self.coord_df[self.coord_df.type == 'street_segment'].frame])
            edgelist = list(self.G.edges())
            edge_pos = np.asarray([(self.pos[e[0]], self.pos[e[1]]) for
                                   e in edgelist])
            self.edge_collection = LineCollection(edge_pos)
            self.edge_collection.set_zorder(1)  # edges go behind nodes
            self.ax[1].scatter(self.corners[:, 0], self.corners[:, 1],
                               c='#fde724')
            self.ax[1].scatter(self.streets[:, 0], self.streets[:, 1],
                               c='#79d151')
            self.ax[1].add_collection(self.edge_collection)
        angle_adj = 0
        agent_loc = self.pos[self.agent_loc]
        agent_dir = self.agent_dir - angle_adj
        goal_loc = self.pos[self.goal_idx]
        goal_dir = self.goal_dir - angle_adj
        print("Agent Dir:", agent_dir)
        print("Goal Dir:", self.goal_dir)
        self.agent_point = self.ax[1].plot(agent_loc[0], agent_loc[1],
                                           color='b', marker='o')
        self.ax[1].plot(goal_loc[0], goal_loc[1], color='r', marker='o')
        self.ax[1].arrow(
            goal_loc[0], goal_loc[1], 5*math.cos(math.radians(goal_dir)),
            5 * math.sin(math.radians(goal_dir)), length_includes_head=True,
            head_width=2.0, head_length=2.0, color='r')
        self.agent_arrow = self.ax[1].arrow(
            agent_loc[0], agent_loc[1], 5 * math.cos(math.radians(agent_dir)),
            5 * math.sin(math.radians(agent_dir)), length_includes_head=True,
            head_width=2.0, head_length=2.0, color='b')
        self.ax[0].set_data(utils.denormalize_image(img.transpose()))
        plt.pause(0.01)
        self.agent_point[0].remove()
        self.agent_arrow.remove()
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
