from __future__ import print_function, division
import enum
import numpy as np
from matplotlib import pyplot as plt
import cv2
from SEVN_gym.envs.SEVN_base import SEVNBase
from SEVN_gym.envs import utils
from SEVN_gym.envs.utils import ActionsWithNOOP


class SEVNPlay(SEVNBase):

    def __init__(self, obs_shape=(84, 84, 3), use_image_obs=True,
                 use_gps_obs=False, use_visible_text_obs=True,
                 use_full=False, reward_type=None, high_res=False):
        super(SEVNPlay, self).__init__(obs_shape, use_image_obs, use_gps_obs,
                                       use_visible_text_obs, use_full,
                                       reward_type)
        self.total_reward = 0
        self.prev_rel_gps = [0, 0, 0, 0]
        self.high_res = high_res
        self.bad_indices = []

    def step(self, a):
        done = False
        action = self._action_set(a)
        image, x, w = self._get_image()
        visible_text = self._get_visible_text(x, w)
        if not action == ActionsWithNOOP.NOOP:
            for text in visible_text['house_numbers']:
                if type(text) == str:
                    print('House number: ' + text)
            for text in visible_text['street_names']:
                if type(text) == str:
                    print('Street name: ' + text)
        if action == ActionsWithNOOP.FORWARD:
            self.transition()
        elif action == ActionsWithNOOP.DONE:
            done = True
        else:
            self.turn(action)

        reward = self.compute_reward(x, {}, done)
        self.agent_gps = utils.sample_gps(self.coord_df.loc[self.agent_loc],
                                          self.x_scale, self.y_scale)
        rel_gps = [self.target_gps[0] - self.agent_gps[0],
                   self.target_gps[1] - self.agent_gps[1],
                   self.target_gps[0], self.target_gps[1]]
        obs = {'image': image, 'mission': self.goal_address,
               'rel_gps': rel_gps, 'visible_text': visible_text}
        self.num_steps_taken += 1
        s = 'obs: '
        for k, v in obs.items():
            if k != 'image':
                s = s + ', ' + str(k) + ': ' + str(v)

        self.total_reward += reward
        if not reward == 0 and not done:
            if not self.prev_rel_gps == rel_gps:
                print('Rel GPS: ' + str(rel_gps[0]) + ', ' + str(rel_gps[1]))
                self.prev_rel_gps = rel_gps
            print('Reward: ' + str(reward))
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

    def reset(self):
        self.num_steps_taken = 0
        self.goal_idx, self.goal_address, self.goal_dir = \
            self.select_goal()
        self.prev_spl = len(self.shortest_path_length())
        self.start_spl = self.prev_spl
        self.agent_gps = utils.sample_gps(self.coord_df.loc[self.agent_loc],
                                          self.x_scale, self.y_scale)
        self.target_gps = utils.sample_gps(self.coord_df.loc[self.goal_idx],
                                           self.x_scale, self.y_scale)
        image, x, w = self._get_image()
        rel_gps = [self.target_gps[0] - self.agent_gps[0],
                   self.target_gps[1] - self.agent_gps[1]]
        return {'image': image,
                'mission': self.goal_address,
                'rel_gps': rel_gps,
                'visible_text': self._get_visible_text(x, w)}

    def compute_reward(self, x, info, done):
        cur_spl = len(self.shortest_path_length())
        if done and self.is_successful_trajectory(x):
            reward = 2.0
        elif done and not self.is_successful_trajectory(x):
            reward = -2.0
        elif self.prev_spl - cur_spl > 0:
            reward = 1
        elif self.prev_spl - cur_spl < 0:
            reward = -1
        else:
            reward = 0.0
        self.prev_spl = cur_spl
        if done:
            print('\n----- Finished -----\nTotal Mission Reward: '
                  + str(reward + self.total_reward)
                  + '\n Success: ' + str(self.is_successful_trajectory(x))
                  + '\n--------------------\n')
        return reward

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

