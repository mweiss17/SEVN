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
