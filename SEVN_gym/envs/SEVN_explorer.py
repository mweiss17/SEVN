from __future__ import print_function, division
from SEVN_gym.envs.SEVN_base import SEVNBase
from SEVN_gym.envs import utils


class SEVNExplorer(SEVNBase):
    def __init__(self, obs_shape=(4, 84, 84), use_image_obs=True,
                 use_gps_obs=False, use_visible_text_obs=True,
                 split="Train", reward_type=None, high_res=False):
        super(SEVNExplorer, self).__init__(obs_shape, use_image_obs, False,
                                           use_visible_text_obs, split,
                                           reward_type, high_res)
        self.seen_house_nums = []
        self.is_explorer = True

    def reset(self):
        self.seen_house_nums = []
        obs = super().reset()
        return obs

    def compute_reward(self, x, action, done, obs):
        reward = 0
        house_numbers = obs['visible_text']['house_numbers']
        house_numbers = utils.convert_house_vec_to_ints(house_numbers)
        for num in house_numbers:
            if num not in self.seen_house_nums:
                reward += 1
                self.seen_house_nums.append(num)
        return reward
