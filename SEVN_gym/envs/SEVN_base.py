from __future__ import print_function, division
import os
import math
import h5py
import zipfile
import numpy as np
import pandas as pd
import dask.array as da
import networkx as nx
import matplotlib.pyplot as plt
import gym
from SEVN_gym.envs.utils import Actions, ACTION_MEANING, get_data_path
from gym import spaces
from matplotlib.collections import LineCollection
from SEVN_gym.envs import utils, wrappers
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# in order to speed up training/NFS file access, you can make 5 copies of
# SEVN_gym/data and call them SEVN_gym/data0 to ...data4. Then this script will
# randomly pick one of those and reduce concurrent access

DATA_PATH = get_data_path()


class SEVNBase(gym.GoalEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 obs_shape=(8, 84, 84),
                 use_image_obs=False,
                 use_gps_obs=False,
                 use_visible_text_obs=False,
                 split="train",
                 reward_type=None,
                 continuous=False):

        print(f'Booting environment from {DATA_PATH} with shaped reward,' +
              f' image_obs: {use_image_obs}, gps: {use_gps_obs},' +
              f' visible_text: {use_visible_text_obs}')

        # Initialize environment
        self.viewer = None
        self.high_res = False
        self.continuous = continuous
        self.use_image_obs = use_image_obs
        self.use_gps_obs = use_gps_obs
        self.use_visible_text_obs = use_visible_text_obs
        self.reward_type = reward_type
        self.needs_reset = True
        self.plot_ready = False
        self._action_set = Actions
        if not self.continuous:
            self.action_space = spaces.Discrete(len(self._action_set))
        else:
            ## pseudo-continuous action. This is jsut a shallow wrapper to use TD3/SAC

            ## forward/backward and left/right, but backward isn't allowed, so forward/noop
            ## and going forward takes priority over turning
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.float32)
        self.agent_dir = 0
        self.num_steps_taken = 0
        self.prev_spl = 100000
        self.goal_hn = -1
        self.SMALL_TURN_DEG = 22.5
        self.BIG_TURN_DEG = 67.5
        self.last_x = None

        # Load data
        if not os.path.isfile(os.path.join(DATA_PATH,'images.hdf5')) \
                or not os.path.isfile(os.path.join(DATA_PATH, 'graph.pkl')) \
                or not os.path.isfile(os.path.join(DATA_PATH, 'label.hdf5')) \
                or not os.path.isfile(os.path.join(DATA_PATH, 'coord.hdf5')):
            # zipfile.ZipFile(at.get("b9e719976cdedb94a25d2f162b899d5f0e711fe0", datastore=DATA_PATH)) \
            #     .extractall(DATA_PATH)
            zipfile.ZipFile(os.path.join(DATA_PATH,
                                         'dataset.zip')).extractall(DATA_PATH)
        f = h5py.File(os.path.join(DATA_PATH, 'images.hdf5'), 'r')
        self.images = f["images"]
        self.frame_key = {int(k): i for i, k in enumerate(f['frames'][:])}
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

        # Set data-dependent variables
        self.max_num_steps = \
            self.coord_df[self.coord_df.type == 'street_segment'].groupby('group').count().max().iloc[0]
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

    def turn(self, action):
        # Modify agent heading
        action = self._action_set(action)
        if action == Actions.LEFT_BIG:
            self.agent_dir += self.BIG_TURN_DEG
        if action == Actions.LEFT_SMALL:
            self.agent_dir += self.SMALL_TURN_DEG
        if action == Actions.RIGHT_SMALL:
            self.agent_dir -= self.SMALL_TURN_DEG
        if action == Actions.RIGHT_BIG:
            self.agent_dir -= self.BIG_TURN_DEG
        self.agent_dir = utils.norm_angle(self.agent_dir)

    def select_goal(self, same_segment=True):
        goals = self.label_df.loc[self.label_df['is_goal'] == True]
        if same_segment:
            frames = self.coord_df[
                (self.coord_df.type == 'street_segment')
                & self.coord_df.index.isin(goals.index)
                & ~self.coord_df.index.isin(self.bad_indices)].index
            goals_on_street_segment = goals[goals.index.isin(frames)]
            goal = goals_on_street_segment.loc[np.random.choice(
                goals_on_street_segment.index.values.tolist())]
            if len(goal.shape) > 1:
                goal = goal.iloc[np.random.randint(len(goal))]
            segment_group = self.coord_df[self.coord_df.index ==
                                          goal.name].group.iloc[0]
            segment_panos = self.coord_df[
                (self.coord_df.group == segment_group)
                & (self.coord_df.type == 'street_segment')]
        else:
            goal = goals.loc[np.random.choice(goals.index.values.tolist())]
        self.goal_hn = goal.house_number
        pano_rotation = utils.norm_angle(self.coord_df.loc[goal.name].angle)
        label = self.label_df.loc[goal.name]
        if isinstance(label, pd.DataFrame):
            label = label[label.is_goal].iloc[np.random.choice(
                label[label.is_goal].shape[0])]
        label_dir = (224 - (label.x_min + label.x_max) / 2) * 360 / 224 - 180
        goal_dir = utils.norm_angle(label_dir + pano_rotation)
        self.agent_dir = self.SMALL_TURN_DEG * np.random.choice(range(-8, 8))
        self.agent_loc = np.random.choice(segment_panos.index.unique())
        goal_address = {
            'house_numbers':
                utils.convert_house_numbers(self.goal_hn),
            'street_names':
                utils.convert_street_name(goal.street_name,
                                          self.all_street_names)
        }
        return goal.name, goal_address, goal_dir

    def transition(self):
        '''
        This function calculates the angles to the other panos then
        transitions to the one that is closest to the agent's current direction
        '''
        neighbors = {}
        for n in [edge[1] for edge in list(self.G.edges(self.agent_loc))]:
            neighbors[n] = np.abs(
                utils.norm_angle(
                    utils.get_angle_between_nodes(self.G, self.agent_loc, n) -
                    self.agent_dir))

        if neighbors[min(neighbors, key=neighbors.get)] > 45:
            return

        self.agent_loc = min(neighbors, key=neighbors.get)
        self.moved_forward = True

    def compute_reward(self, x, action, done):
        assert action in Actions

        # if action == Actions.NOOP:
        #     return -0.1  # we don't wanna just stand around

        if action == Actions.FORWARD:
            # If action was forward
            prev_spl = self.prev_spl
            if prev_spl == 1 or self.prev_sp[1] != self.agent_loc:
                ## if the agent did NOT go to the next correct node
                sp = nx.shortest_path(
                    self.G, self.agent_loc, target=self.goal_idx)
                self.prev_sp = sp
                self.prev_spl = len(sp)
            else:
                ## if the agent moved forward
                self.prev_sp = self.prev_sp[1:]
                self.prev_spl = len(self.prev_sp)

            if self.prev_spl < prev_spl:
                self.min_spl = self.prev_spl
                return 1
            return -1

        if action in [
                Actions.LEFT_SMALL, Actions.LEFT_BIG, Actions.RIGHT_SMALL,
                Actions.RIGHT_BIG
        ]:
            ## If action was turn

            if self.prev_spl == 1:
                ## Goal node:
                target_dir = self.goal_dir
            else:
                target_dir = utils.angle_to_node(self.G, self.agent_loc,
                                                 self.prev_sp[1],
                                                 self.SMALL_TURN_DEG)

            angle = utils.smallest_angle(self.old_agent_dir, target_dir)
            new_angle = utils.smallest_angle(self.agent_dir, target_dir)

            ## if we just moved forward than our base orientation is our new baseline
            if self.moved_forward:
                self.moved_forward = False
                self.min_angle = angle

            assert angle != new_angle  # just making sure

            # this is to prevent exploitation by only incentivizing approaching the goal,
            # not going too far and coming back
            multiplier = 1
            # if self.prev_spl > self.min_spl: # prev_spl is current_spl
            #     multiplier = 0

            if np.abs(new_angle) < np.abs(self.min_angle):
                ## nice, we found an angle that is closer to the goal
                self.min_angle = new_angle
                ## this min_angle reset upon walking forward
                return .1 * multiplier
            elif np.abs(new_angle) > np.abs(angle):
                ## colder
                return -.2
            elif np.abs(new_angle) < np.abs(angle):
                ## warmer, but we've done better before, so no +1
                return 0

    def step(self, a):
        done = False
        was_successful_trajectory = False
        oracle = False
        reward = -1

        self.num_steps_taken += 1

        # if self.continuous:
        #     a = continuous2discrete(a)

        action = self._action_set(a)
        if oracle:
            action = next(iter(self.shortest_path_length()), None)
        if self.last_x is None:
            raise Exception("Must run `env.reset()` once before first step")

        self.old_agent_dir = self.agent_dir

        if action == Actions.FORWARD:
            self.transition()
        # elif action == Actions.NOOP:
        #     pass  # no change to position
        else:
            self.turn(action)

        image, x, w = self._get_image()
        reward = self.compute_reward(x, action, done)

        if self.is_successful_trajectory(x):
            done = True
            was_successful_trajectory = True
        elif self.num_steps_taken >= self.max_num_steps and done is False:
            done = True
        visible_text = self._get_visible_text(x, w)
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
        obs = wrappers.wrap_obs(obs, self.use_gps_obs,
                                self.use_visible_text_obs, self.use_image_obs,
                                True, self.num_streets)

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
        normed_ang = ((agent_dir - pano_rotation) % 360) / 360
        w = obs_shape[1]
        y = img.shape[0] - obs_shape[1]
        h = obs_shape[2]
        x = int(img.shape[1] - (
            (normed_ang * img.shape[1]) + img.shape[1] / 2 + obs_shape[1] / 2) %
                img.shape[1])
        img = img.transpose()
        if (x + w) % img.shape[1] != (x + w):
            res_img = np.zeros((3, 84, 84))
            offset = img.shape[1] - (x % img.shape[1])
            res_img[:, :offset] = img[y:y + h, x:x + offset]
            res_img[:, offset:] = img[y:y + h, :(x + w) % img.shape[1]]
        else:
            res_img = img[:, x:x + w]

        self.last_x = x
        return res_img, x, w

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
        # TODO: remove this when we no longer need profiling
        self.diff_a = []
        self.diff_b = []
        self.diff_c = []
        self.diff_d = []
        self.diff_e = []
        self.diff_f = []

        self.needs_reset = False
        self.min_angle = np.inf
        self.min_spl = np.inf
        self.moved_forward = False
        self.num_steps_taken = 0
        self.goal_idx, self.goal_address, self.goal_dir = \
            self.select_goal(same_segment=True)
        self.prev_sp = nx.shortest_path(
            self.G, self.agent_loc, target=self.goal_idx)
        self.prev_spl = len(self.prev_sp)
        self.agent_gps = utils.sample_gps(self.coord_df.loc[self.agent_loc],
                                          self.x_scale, self.y_scale)
        self.target_gps = utils.sample_gps(self.coord_df.loc[self.goal_idx],
                                           self.x_scale, self.y_scale)
        image, x, w = self._get_image()
        rel_gps = [
            self.target_gps[0] - self.agent_gps[0],
            self.target_gps[1] - self.agent_gps[1], self.target_gps[0],
            self.target_gps[1]
        ]
        obs = {
            'image': image,
            'mission': self.goal_address,
            'rel_gps': rel_gps,
            'visible_text': self._get_visible_text(x, w)
        }
        obs = wrappers.wrap_obs(obs, self.use_gps_obs,
                                self.use_visible_text_obs, self.use_image_obs,
                                True, self.num_streets)
        return obs

    def angles_to_turn(self, cur, target):
        angle = utils.smallest_angle(cur, target)
        turns = []
        if (np.sign(angle)) == 1:
            big_turns = int(angle / self.BIG_TURN_DEG)
            turns.extend([Actions.LEFT_BIG for i in range(big_turns)])
            cur = utils.norm_angle(cur + big_turns * self.BIG_TURN_DEG)
            small_turns = int(
                (angle - big_turns * self.BIG_TURN_DEG) / self.SMALL_TURN_DEG)
            turns.extend([Actions.LEFT_SMALL for i in range(small_turns)])
            cur = utils.norm_angle(cur + small_turns * self.SMALL_TURN_DEG)
        else:
            angle = np.abs(angle)
            big_turns = int(angle / self.BIG_TURN_DEG)
            turns.extend([Actions.RIGHT_BIG for i in range(big_turns)])
            cur = utils.norm_angle(cur - big_turns * self.BIG_TURN_DEG)
            small_turns = int(
                (angle - big_turns * self.BIG_TURN_DEG) / self.SMALL_TURN_DEG)
            turns.extend([Actions.RIGHT_SMALL for i in range(small_turns)])
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
                target_dir = utils.angle_to_node(self.G, node, path[idx + 1],
                                                 self.SMALL_TURN_DEG)
                new_action, final_dir = self.angles_to_turn(cur_dir, target_dir)

                actions.extend(new_action)
                cur_dir = final_dir
                actions.append(Actions.FORWARD)

            else:
                new_action, final_dir = self.angles_to_turn(
                    cur_dir, self.goal_dir)
                actions.extend(new_action)

        return actions

    def is_successful_trajectory(self, x):
        if not self.agent_loc == self.goal_idx:
            return False
        try:
            label = self.label_df.loc[self.agent_loc, [
                'frame', 'obj_type', 'house_number', 'x_min', 'x_max'
            ]]
            if isinstance(label, pd.DataFrame):
                label = label[(label.house_number == self.goal_hn)
                              & (label.obj_type == 'door')]
            label = label.iloc[np.random.randint(len(label))]
            if label.empty:
                return False
            if x < label.x_min and x + 84 > label.x_max:
                return x < label.x_min and x + 84 > label.x_max
        except (KeyError, IndexError, AttributeError, ValueError) as e:
            return False

    def render(self, mode='human', clear=False, first_time=False):
        img, x, w = self._get_image()

        if first_time or not self.plot_ready:
            plt.ion()
            self.fig, self.ax = plt.subplots(nrows=1, ncols=2)
            self.ax[0] = self.ax[0].imshow(
                np.zeros((84, 84, 3)),
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
        angle_adj = 0
        agent_loc = self.pos[self.agent_loc]
        agent_dir = self.agent_dir - angle_adj
        goal_loc = self.pos[self.goal_idx]
        goal_dir = self.goal_dir - angle_adj
        print("Agent Dir:", agent_dir)
        print("Goal Dir:", self.goal_dir)
        self.agent_point = self.ax[1].plot(
            agent_loc[0], agent_loc[1], color='b', marker='o')
        self.ax[1].plot(goal_loc[0], goal_loc[1], color='r', marker='o')
        self.ax[1].arrow(
            goal_loc[0],
            goal_loc[1],
            5 * math.cos(math.radians(goal_dir)),
            5 * math.sin(math.radians(goal_dir)),
            length_includes_head=True,
            head_width=2.0,
            head_length=2.0,
            color='r')
        self.agent_arrow = self.ax[1].arrow(
            agent_loc[0],
            agent_loc[1],
            5 * math.cos(math.radians(agent_dir)),
            5 * math.sin(math.radians(agent_dir)),
            length_includes_head=True,
            head_width=2.0,
            head_length=2.0,
            color='b')
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
