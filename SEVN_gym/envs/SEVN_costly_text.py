from __future__ import print_function, division

from SEVN_gym.envs import utils, wrappers
from SEVN_gym.envs.SEVN_base import SEVNBase
from SEVN_gym.envs.utils import ActionsWithRead as Actions


class SEVNCostlyText(SEVNBase):
    def __init__(self, obs_shape=(4, 84, 84), use_image_obs=True,
                 use_gps_obs=True, use_visible_text_obs=False,
                 split="Train", reward_type=None, high_res=False):
        self._action_set = Actions
        import pdb; pdb.set_trace()
        super(SEVNCostlyText, self).__init__(obs_shape=obs_shape, use_image_obs=use_image_obs,
                                             use_gps_obs=use_gps_obs, use_visible_text_obs=use_visible_text_obs,
                                             split=split, reward_type=reward_type, high_res=high_res)

    def step(self, a):
        action = self._action_set(a)
        self.use_visible_text_obs = False
        if action == Actions.READ:
            self.use_visible_text_obs = True
        return super().step(a)

    def compute_reward(self, x, action, done, obs=None):
        reward = super().compute_reward(x, action, done, obs)
        if action == self._action_set.READ:
            reward = -0.1
        return reward
