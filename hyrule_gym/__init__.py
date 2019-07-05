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

# for loop w/ mini and full dataset Hyrule-{}-FullShaped-v1 (use f notation)
for dataset in ["Full", "Mini"]:
    for modality in ["All", "NoImg", "NoGPS", "ImgOnly"]:
        for reward in ["Shaped", "Sparse"]:
            id = f"Hyrule-{dataset}-{modality}-{reward}-v1"
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
                entry_point=f'hyrule_gym.envs:HyruleEnv{reward}',
                kwargs={'use_image_obs': use_image_obs, 'use_gps_obs': use_gps_obs, 'use_visible_text_obs': use_visible_text_obs, "use_full": use_full},
            )
