from gym.envs.registration import register


register(
    id='SEVN-Explorer-v1',
    entry_point='SEVN_gym.envs:SEVNExplorer',
    kwargs={},
    max_episode_steps=10000,
)

register(
    id='SEVN-CostlyText-v1',
    entry_point='SEVN_gym.envs:SEVNCostlyText',
    kwargs={},
    max_episode_steps=10000,
)

register(
    id='SEVN-Test-DecreasingReward-v1',
    entry_point='SEVN_gym.envs:SEVNDecreasingReward',
    kwargs={},
    max_episode_steps=10000,
)

register(
    id='SEVN-Train-DecreasingReward-v1',
    entry_point='SEVN_gym.envs:SEVNDecreasingReward',
    kwargs={'split': 'Train'},
    max_episode_steps=10000,
)

register(
    id='SEVN-Test-NoisyGPS-1-v1',
    entry_point='SEVN_gym.envs:SEVNNoisyGPS',
    kwargs={'split': 'Test', "noise_scale": 1},
    max_episode_steps=10000,
)

register(
    id='SEVN-Test-NoisyGPS-5-v1',
    entry_point='SEVN_gym.envs:SEVNNoisyGPS',
    kwargs={'split': 'Test', "noise_scale": 5},
    max_episode_steps=10000,
)

register(
    id='SEVN-Test-NoisyGPS-25-v1',
    entry_point='SEVN_gym.envs:SEVNNoisyGPS',
    kwargs={'split': 'Test', "noise_scale": 25},
    max_episode_steps=10000,
)

register(
    id='SEVN-Test-NoisyGPS-100-v1',
    entry_point='SEVN_gym.envs:SEVNNoisyGPS',
    kwargs={'split': 'Test', "noise_scale": 100},
    max_episode_steps=10000,
)

register(
    id='SEVN-Play-v1',
    entry_point='SEVN_gym.envs:SEVNPlay',
    kwargs={},
    max_episode_steps=10000,
)

register(
    id='SEVN-Play-HighRes-v1',
    entry_point='SEVN_gym.envs:SEVNPlay',
    kwargs={'high_res': True},
    max_episode_steps=10000,
)


for split in ['Train', 'Test', 'AllData']:
    for modality in ['AllObs', 'NoImg', 'NoGPS', 'ImgOnly']:
        for reward in ['Shaped', 'Sparse']:
            for action in ['','-Continuous']:
                id = f'SEVN-{split}-{modality}-{reward}{action}-v1'
                use_image_obs = False
                use_visible_text_obs = False
                use_gps_obs = False
                continuous = False
                if modality == 'AllObs':
                    use_image_obs = True
                    use_visible_text_obs = True
                    use_gps_obs = True
                elif modality == 'NoImg':
                    use_visible_text_obs = True
                    use_gps_obs = True
                elif modality == 'NoGPS':
                    use_visible_text_obs = True
                    use_image_obs = True
                elif modality == 'ImgOnly':
                    use_image_obs = True
                if action != '':
                    continuous = True
                register(
                    id=id,
                    entry_point=f'SEVN_gym.envs:SEVNBase',
                    kwargs={'use_image_obs': use_image_obs,
                            'use_gps_obs': use_gps_obs,
                            'use_visible_text_obs': use_visible_text_obs,
                            'split': split,
                            'reward_type': reward,
                            'continuous': continuous
                            },
                    max_episode_steps=255,
                )
