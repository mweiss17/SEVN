import numpy as np


def wrap_obs(obs, use_gps_obs, use_visible_text_obs, use_image_obs, use_goal,
             num_streets):
    coord_holder = np.zeros((1, 84, 84), dtype=np.float32)

    if use_gps_obs:
        coord_holder[0, 0, :4] = obs['rel_gps']

    if use_visible_text_obs:
        coord_holder[0, 1, :2 * num_streets] = \
            obs['visible_text']['street_names']
        coord_holder[0, 2, :] = \
            obs['visible_text']['house_numbers'][:84]
        coord_holder[0, 3, :36] = obs['visible_text']['house_numbers'][84:120]
    if not use_image_obs:
        obs['image'] = np.zeros((3, 84, 84))

    if use_goal:
        coord_holder[0, 4, :40] = obs['mission']['house_numbers']
        coord_holder[0, 4, 40:40 + num_streets] = \
            obs['mission']['street_names']
    out = np.concatenate((obs['image'], coord_holder), axis=0)
    return out


def unwrap_obs(obs, use_gps_obs, use_visible_text_obs, use_image_obs, use_goal,
               num_streets):
    data = obs[3, :, :]

    out = {}
    if use_gps_obs:
        out['rel_gps'] = data[0, :4]

    if use_visible_text_obs:
        out['visible_street_names'] = data[1, :2 * num_streets].reshape(
            (2, num_streets))
        out['visible_house_numbers'] = np.hstack(
            (data[2, :], data[3, :36])).reshape((3, 4, 10))  # [84:120]

    # if not use_image_obs:
    #     obs['image'] = np.zeros((3, 84, 84))

    if use_goal:
        out['goal_house_numbers'] = data[4, :40].reshape((4, 10))
        out['goal_street_names'] = data[4, 40:40 + num_streets]

    return out
