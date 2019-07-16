import numpy as np

def wrap_obs(obs, use_gps_obs, use_visible_text_obs, use_image_obs, use_goal, num_streets):
    coord_holder = np.zeros((1, 84, 84), dtype=np.float32)

    if use_gps_obs:
        coord_holder[0, 0, :4] = obs['rel_gps']

    if use_visible_text_obs:
        coord_holder[0, 1, :2 * num_streets] = obs['visible_text']['street_names']
        coord_holder[0, 2, :] = obs['visible_text']['house_numbers'][:84]
        coord_holder[0, 3, :36] = obs['visible_text']['house_numbers'][84:120]
    if not use_image_obs:
        obs['image'] = np.zeros((3, 84, 84))

    if use_goal:
        coord_holder[0, 4, :40] = obs['mission']['house_numbers']
        coord_holder[0, 4, 40:40 + num_streets] = obs['mission']["street_names"]
    out = np.concatenate((obs['image'], coord_holder), axis=0)
    return out
