import os

from gym.envs.registration import register

register(
    id='Hyrule-v1',
    entry_point='hyrule_gym.envs:HyruleEnv',
    kwargs={'obs_type': 'image'},
    max_episode_steps=10000,
)

register(
    id='Hyrule-done-v1',
    entry_point='hyrule_gym.envs:HyruleEnv',
    kwargs={'obs_type': 'image', 'can_done': True},
    max_episode_steps=10000,
)

register(
    id='Hyrule-noop-v1',
    entry_point='hyrule_gym.envs:HyruleEnv',
    kwargs={'obs_type': 'image', 'can_noop': True},
    max_episode_steps=10000,
)

register(
    id='Hyrule-FullShaped-v1',
    entry_point='hyrule_gym.envs:HyruleEnvShaped',
    kwargs={'use_image_obs': True, 'use_gps_obs': True, 'use_visible_text_obs': True},
    max_episode_steps=10000,
)

register(
    id='Hyrule-NoImgShaped-v1',
    entry_point='hyrule_gym.envs:HyruleEnvShaped',
    kwargs={'use_image_obs': False, 'use_gps_obs': True, 'use_visible_text_obs': True},
    max_episode_steps=10000,
)

register(
    id='Hyrule-NoGPSShaped-v1',
    entry_point='hyrule_gym.envs:HyruleEnvShaped',
    kwargs={'use_image_obs': True, 'use_gps_obs': False, 'use_visible_text_obs': True},
    max_episode_steps=10000,
)

register(
    id='Hyrule-ImgOnlyShaped-v1',
    entry_point='hyrule_gym.envs:HyruleEnvShaped',
    kwargs={'use_image_obs': True, 'use_gps_obs': False, 'use_visible_text_obs': False},
    max_episode_steps=10000,
)
