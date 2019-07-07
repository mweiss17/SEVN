import os

from gym.envs.registration import register


# register(
#     id='SEVN-noop-v1',
#     entry_point='SEVN_gym.envs:SEVNEnv',
#     kwargs={'obs_type': 'image', 'can_noop': True},
#     max_episode_steps=10000,
# )

register(
    id='SEVN-Explorer-v1',
    entry_point='SEVN_gym.envs:SEVNExplorer',
    kwargs={},
    max_episode_steps=10000,
)

register(
    id='SEVN-Mini-DecreasingReward-v1',
    entry_point='SEVN_gym.envs:SEVNDecreasingReward',
    kwargs={},
    max_episode_steps=10000,
)

register(
    id='SEVN-Full-DecreasingReward-v1',
    entry_point='SEVN_gym.envs:SEVNDecreasingReward',
    kwargs={"use_full": True},
    max_episode_steps=10000,
)

register(
    id='SEVN-Mini-NoisyGPS-1-v1',
    entry_point='SEVN_gym.envs:SEVNNoisyGPS',
    kwargs={"use_full": False, "noise_scale": 1},
    max_episode_steps=10000,
)

register(
    id='SEVN-Mini-NoisyGPS-5-v1',
    entry_point='SEVN_gym.envs:SEVNNoisyGPS',
    kwargs={"use_full": False, "noise_scale": 5},
    max_episode_steps=10000,
)

register(
    id='SEVN-Play-v1',
    entry_point='SEVN_gym.envs:SEVNPlay',
    kwargs={},
    max_episode_steps=10000,
)


for dataset in ["Full", "Mini"]:
    for modality in ["All", "NoImg", "NoGPS", "ImgOnly"]:
        for reward in ["Shaped", "Sparse"]:
            id = f"SEVN-{dataset}-{modality}-{reward}-v1"
            use_full = False
            if dataset == "Full":
                use_full = True
            use_image_obs = False
            use_visible_text_obs = False
            use_gps_obs = False
            if modality == "All":
                use_image_obs = True
                use_visible_text_obs = True
                use_gps_obs = True
            elif modality == "NoImg":
                use_visible_text_obs = True
                use_gps_obs = True
            elif modality == "NoGPS":
                use_visible_text_obs = True
                use_image_obs = True
            elif modality == "ImgOnly":
                use_image_obs = True
            register(
                id=id,
                entry_point=f'SEVN_gym.envs:SEVNBase',
                kwargs={'use_image_obs': use_image_obs, 'use_gps_obs': use_gps_obs, 'use_visible_text_obs': use_visible_text_obs, "use_full": use_full, "reward_type": reward},
            )
