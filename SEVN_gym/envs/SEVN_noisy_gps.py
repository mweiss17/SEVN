from __future__ import print_function, division
from SEVN_gym.envs.SEVN_base import SEVNBase


class SEVNNoisyGPS(SEVNBase):
    def __init__(self, obs_shape=(8, 84, 84), use_image_obs=True,
                 use_gps_obs=True, use_visible_text_obs=True, split="Train",
                 reward_type=None, noise_scale=0.0):
        super(SEVNNoisyGPS, self).__init__(obs_shape, use_image_obs,
                                           use_gps_obs, use_visible_text_obs,
                                           split, reward_type)
        self.noise_scale = noise_scale
