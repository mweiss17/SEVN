from __future__ import print_function, division
import enum
import math
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import gym
import gzip
from gym import spaces
from SEVN_gym.data import _ROOT
from SEVN_gym.envs import utils, wrappers


ACTION_MEANING = {
    0: 'LEFT_BIG',
    1: 'LEFT_SMALL',
    2: 'FORWARD',
    3: 'RIGHT_SMALL',
    4: 'RIGHT_BIG',
    5: 'DONE',
    6: 'NOOP',
}

class SEVNBase(gym.GoalEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    class Actions(enum.IntEnum):
        LEFT_BIG = 0
        LEFT_SMALL = 1
        FORWARD = 2
        RIGHT_SMALL = 3
        RIGHT_BIG = 4

    def __init__(self, obs_shape=(4, 84, 84), use_image_obs=False, use_gps_obs=False, use_visible_text_obs=False, use_full=False, reward_type=None):
        path = "/SEVN/processed/" if use_full else "/SEVN-mini/processed/"
        path = _ROOT + path

        print(f"Booting environment from {path} with shaped reward, image_obs: {use_image_obs}, gps: {use_gps_obs}, visible_text: {use_visible_text_obs}")
        self.viewer = None
        self.high_res = False
        self.use_image_obs = use_image_obs
        self.use_gps_obs = use_gps_obs
        self.use_visible_text_obs = use_visible_text_obs
        self.reward_type = reward_type
        self.needs_reset = True
        self._action_set = SEVNBase.Actions
        self.action_space = spaces.Discrete(len(self._action_set))
        # spaces.dict goes here
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.float32)
        f = gzip.GzipFile(path + "images.pkl.gz", "r")
        self.images_df = pickle.load(f)
        f.close()
        self.meta_df = pd.read_hdf(path + "meta.hdf5", key='df', mode='r')
        self.G = nx.read_gpickle(path + "graph.pkl")
        self.max_num_steps = self.meta_df[self.meta_df.type == "street_segment"].groupby(self.meta_df.group).count().max().frame
        self.all_street_names = self.meta_df.street_name.dropna().unique()
        self.num_streets = self.all_street_names.size
        self.x_scale = self.meta_df.x.max() - self.meta_df.x.min()
        self.y_scale = self.meta_df.y.max() - self.meta_df.y.min()
        self.agent_loc = np.random.choice(self.meta_df.frame)
        self.agent_dir = 0
        self.num_steps_taken = 0
        self.prev_spl = 100000
        self.goal_id = -1

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
        self.agent_dir = utils.norm_angle(self.agent_dir)


    def get_angle_between_nodes(self, n1, n2):
        x = self.G.nodes[n2]['coords'][0] - self.G.nodes[n1]['coords'][0]
        y = self.G.nodes[n2]['coords'][1] - self.G.nodes[n1]['coords'][1]
        angle = (math.atan2(y, x) * 180 / np.pi)
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
        pano_rotation = utils.norm_angle(self.meta_df.loc[goal_idx].angle.iloc[0])
        label_dir = (224 - (label.x_min.values[0] + label.x_max.values[0]) / 2) * 360 / 224 - 180
        goal_dir = utils.norm_angle(label_dir + pano_rotation)
        self.agent_dir = 22.5 * np.random.choice(range(-8, 8))
        self.agent_loc = np.random.choice(segment_panos.frame.unique())
        goal_address = {"house_numbers": utils.convert_house_numbers(int(goal.house_number.iloc[0])),
                        "street_names": utils.convert_street_name(goal.street_name.iloc[0], self.all_street_names)}
        return goal_idx, goal_address, goal_dir

    def transition(self):
        """
        This function calculates the angles to the other panos
        then transitions to the one that is closest to the agent's current direction
        """
        neighbors = {}
        for n in [edge[1] for edge in list(self.G.edges(self.agent_loc))]:
            neighbors[n] = np.abs(utils.norm_angle(self.get_angle_between_nodes(self.agent_loc, n) - self.agent_dir))

        if neighbors[min(neighbors, key=neighbors.get)] > 22.5:
            return

        self.agent_loc = min(neighbors, key=neighbors.get)

    def step(self, a):
        done = False
        was_successful_trajectory = False
        oracle = True

        reward = 0.0
        self.num_steps_taken += 1
        action = self._action_set(a)
        if oracle:
            action = next(iter(self.shortest_path_length()), None)
        image, x, w = self._get_image()
        visible_text = self.get_visible_text(x, w)
        try:
            if self.is_successful_trajectory(x):
                done = True
                was_successful_trajectory = True
            elif self.num_steps_taken >= self.max_num_steps and done == False:
                done = True
            elif action == self.Actions.FORWARD:
                self.transition()
            else:
                self.turn(action)
        except Exception:
            import pdb; pdb.set_trace()
            self.is_successful_trajectory(x)

        reward = self.compute_reward(x, {}, done)

        self.agent_gps = utils.sample_gps(self.meta_df.loc[self.agent_loc], self.x_scale, self.y_scale)
        rel_gps = [self.target_gps[0] - self.agent_gps[0], self.target_gps[1] - self.agent_gps[1],
                   self.target_gps[0], self.target_gps[1]]
        obs = {"image": image, "mission": self.goal_address, "rel_gps": rel_gps, "visible_text": visible_text}
        obs = wrappers.wrap_obs(obs, self.use_gps_obs, self.use_visible_text_obs, self.use_image_obs, True, self.num_streets)

        info = {}
        if done:
            info["was_successful_trajectory"] = was_successful_trajectory
            self.needs_reset = True

        return obs, reward, done, info


    def _get_image(self):
        img = self.images_df[self.meta_df.loc[self.agent_loc, 'frame'][0]]
        img = utils.normalize_image(img)
        obs_shape = self.observation_space.shape

        pano_rotation = self.meta_df.loc[self.agent_loc, 'angle'][0]
        agent_dir = utils.norm_angle_360(self.agent_dir)
        normed_ang = ((agent_dir - pano_rotation) % 360)/360
        w = obs_shape[1]
        y = img.shape[0] - obs_shape[1]
        h = obs_shape[2]
        x = int(img.shape[1] - ((normed_ang * img.shape[1]) + img.shape[1]/2 + obs_shape[1]/2) % img.shape[1])
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
        subset = self.meta_df.loc[self.agent_loc, ["house_number", "street_name", "obj_type", "x_min", "x_max"]]
        house_numbers, street_names = utils.extract_text(x, w, subset, self.high_res)
        house_numbers = [utils.convert_house_numbers(num) for num in house_numbers]
        street_names = [utils.convert_street_name(name, self.all_street_names) for name in street_names]
        visible_text = utils.stack_text(house_numbers, street_names, self.num_streets)
        return visible_text

    def reset(self):
        self.needs_reset = False
        self.num_steps_taken = 0
        self.goal_idx, self.goal_address, self.goal_dir = self.select_goal(same_segment=True)
        self.prev_spl = len(self.shortest_path_length())
        self.start_spl = self.prev_spl
        self.agent_gps = utils.sample_gps(self.meta_df.loc[self.agent_loc], self.x_scale, self.y_scale)
        self.target_gps = utils.sample_gps(self.meta_df.loc[self.goal_idx], self.x_scale, self.y_scale)
        image, x, w = self._get_image()
        rel_gps = [self.target_gps[0] - self.agent_gps[0], self.target_gps[1] - self.agent_gps[1],
                   self.target_gps[0], self.target_gps[1]]
        obs = {"image": image, "mission": self.goal_address, "rel_gps": rel_gps, "visible_text": self.get_visible_text(x, w)}
        obs = wrappers.wrap_obs(obs, self.use_gps_obs, self.use_visible_text_obs, self.use_image_obs, True, self.num_streets)
        return obs

    def angles_to_turn(self, cur, target):
        angle = utils.smallest_angle(cur, target)
        turns = []
        if (np.sign(angle)) == 1:
            big_turns = int(angle / 67.5)
            turns.extend([self.Actions.LEFT_BIG for i in range(big_turns)])
            cur = utils.norm_angle(cur + big_turns * 67.5)
            small_turns = int((angle - big_turns * 67.5) / 22.5)
            turns.extend([self.Actions.LEFT_SMALL for i in range(small_turns)])
            cur = utils.norm_angle(cur + small_turns * 22.5)
        else:
            angle = np.abs(angle)
            big_turns = int(angle / 67.5)
            turns.extend([self.Actions.RIGHT_BIG for i in range(big_turns)])
            cur = utils.norm_angle(cur - big_turns * 67.5)
            small_turns = int((angle - big_turns * 67.5) / 22.5)
            turns.extend([self.Actions.RIGHT_SMALL for i in range(small_turns)])
            cur = utils.norm_angle(cur - small_turns * 22.5)
        return turns, cur

    def shortest_path_length(self):
        # finds a minimal trajectory to navigate to the target pose
        # target_index = self.coords_df[self.coords_df.frame == int(target_node_info['timestamp'] * 30)].index.values[0]
        cur_node = self.agent_loc
        cur_dir = self.agent_dir
        target_node = self.goal_idx
        path = nx.shortest_path(self.G, cur_node, target=target_node)
        actions = []
        for idx, node in enumerate(path):
            if idx + 1 != len(path):
                # target_dir = self.get_angle_between_nodes(node, path[idx + 1])
                target_dir = self.angle_to_node(node, path[idx + 1])
                new_action, final_dir = self.angles_to_turn(cur_dir, target_dir)
                actions.extend(new_action)
                cur_dir = final_dir
                actions.append(self.Actions.FORWARD)
            else:
                new_action, final_dir = self.angles_to_turn(cur_dir, self.goal_dir)
                actions.extend(new_action)
        return actions

    def angle_to_node(self, n1, n2):
        node_dir = self.get_angle_between_nodes(n1, n2)
        neighbors = [edge[1] for edge in list(self.G.edges(n1))]
        neighbor_angles = []
        for neighbor in neighbors:
            neighbor_angles.append(self.get_angle_between_nodes(n1, neighbor))

        dest_nodes = {}
        for direction in [x*22.5 for x in range(-8, 8)]:
            angles = utils.smallest_angles(direction, neighbor_angles)
            min_angle_node = neighbors[angles.index(min(angles))]
            if min(angles) < 22.5:
                dest_nodes[direction] = min_angle_node
            else:
                dest_nodes[direction] = None

        valid_angles = []
        dist = []
        for k, v in dest_nodes.items():
            if v == n2:
                valid_angles.append(k)
                dist.append(np.abs(utils.smallest_angle(k, node_dir)))

        return valid_angles[dist.index(min(dist))]

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

        if self.reward_type == "Sparse" and reward != 2.0 and reward != -2.0:
            reward = 0
        self.prev_spl = cur_spl
        return reward

    def is_successful_trajectory(self, x):
        subset = self.meta_df.loc[self.agent_loc, ["frame", "obj_type", "house_number", "x_min", "x_max"]]
        label = subset[(subset.house_number == self.goal_id.iloc[0]) & (subset.obj_type == "door")]
        try:
            return x < label.x_min.iloc[0] and x + 84 > label.x_max.iloc[0]
        except Exception:
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
