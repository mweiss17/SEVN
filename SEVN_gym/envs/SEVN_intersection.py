from __future__ import print_function, division
from SEVN_gym.envs.SEVN_base import SEVNBase

class SEVNIntersection(SEVNBase):
    def __init__(self, obs_shape=(4, 84, 84), use_image_obs=True,
                 use_gps_obs=True, use_visible_text_obs=False,
                 split="Train", reward_type=None, high_res=False):
        super(SEVNIntersection, self).__init__(obs_shape=obs_shape, use_image_obs=use_image_obs,
                                             use_gps_obs=use_gps_obs, use_visible_text_obs=use_visible_text_obs,
                                             split=split, reward_type=reward_type, high_res=high_res)
        self.goal_type = "intersection"
