import numpy as np


def wrap_obs(obs, use_gps_obs, use_visible_text_obs, use_image_obs, use_goal, num_streets):
    w = obs['image'].shape[1]

    if use_gps_obs:
        # Relative and Absolute gps (4-D input)
        gps_fm = np.array(obs['rel_gps'], dtype=np.float32)
        gps_fm = np.tile(gps_fm, (w, w // len(gps_fm)))
        gps_fm = np.expand_dims(gps_fm, axis=0)
    else:
        gps_fm = np.zeros((1, w, w), dtype=np.float32)

    if use_visible_text_obs:
        # Visible street signs (2 X num_streets input)
        vsn_fm = np.array([obs['visible_text']['street_names'][:num_streets],
                           obs['visible_text']['street_names'][num_streets:2*num_streets]],
                          dtype=np.float32)
        vsn_fm = np.tile(vsn_fm, (1, w // num_streets))
        vsn_fm = np.pad(vsn_fm, ((0, 0), (0, w % num_streets)), 'constant', constant_values=0)
        vsn_fm = np.tile(vsn_fm, (int(w/2), 1))
        vsn_fm = np.expand_dims(vsn_fm, axis=0)

        # Visible house numbers (3 X 40 input)
        vhn_fm = np.array([obs['visible_text']['house_numbers'][:40],
                           obs['visible_text']['house_numbers'][40:80],
                           obs['visible_text']['house_numbers'][80:120]], dtype=np.float32)
        if w == 84:
            vhn_fm = np.tile(vhn_fm, (1, 2))
            vhn_fm = np.pad(vhn_fm, ((0, 0), (0, 4)), 'constant', constant_values=0)
            vhn_fm = np.tile(vhn_fm, (28, 1))
        else:
            vhn_fm = vhn_fm.flatten()
            vhn_fm = np.tile(vhn_fm, w // 120)
            vhn_fm = np.pad(vhn_fm, int((w % vhn_fm.shape[0]) / 2), 'constant')
            vhn_fm = np.tile(vhn_fm, (1280, 1))
        vhn_fm = np.expand_dims(vhn_fm, axis=0)
    else:
        vsn_fm = np.zeros((1, w, w), dtype=np.float32)
        vhn_fm = np.zeros((1, w, w), dtype=np.float32)

    if not use_image_obs:
        obs['image'] = np.zeros((3, w, w), dtype=np.float32)

    if use_goal:
        ghn_fm = np.array(obs['mission']['house_numbers'], dtype=np.float32)
        if w == 84:
            ghn_fm = np.tile(ghn_fm, (1, 2))
            ghn_fm = np.pad(ghn_fm, ((0, 0), (0, 4)), 'constant', constant_values=0)
            ghn_fm = np.tile(ghn_fm, (84, 1))
        else:
            ghn_fm = np.tile(ghn_fm, w // 40)
            ghn_fm = np.tile(ghn_fm, (1280, 1))
        ghn_fm = np.expand_dims(ghn_fm, axis=0)

        # Goal street signs (num_streets input)
        gsn_fm = np.array(obs['mission']['street_names'], dtype=np.float32)
        if w == 84:
            gsn_fm = np.tile(gsn_fm, (1, 84 // num_streets))
            gsn_fm = np.pad(gsn_fm, ((0, 0), (0, 84 % num_streets)), 'constant', constant_values=0)
            gsn_fm = np.tile(gsn_fm, (84, 1))
        else:
            gsn_fm = np.tile(gsn_fm, (1, w // num_streets))
            gsn_fm = gsn_fm.flatten()
            gsn_fm = np.pad(gsn_fm, (0, int((w % num_streets))), 'constant')
            gsn_fm = np.tile(gsn_fm, (w, 1))
        gsn_fm = np.expand_dims(gsn_fm, axis=0)


    else:
        ghn_fm = np.zeros((1, w, w), dtype=np.float32)
        gsn_fm = np.zeros((1, w, w), dtype=np.float32)
    out = np.concatenate((obs['image'], gps_fm, vsn_fm, vhn_fm, ghn_fm, gsn_fm), axis=0)
    return out


def _wrap_obs(obs, use_gps_obs, use_visible_text_obs, use_image_obs, use_goal,
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


def unwrap_obs(obs, use_gps_obs, use_visible_text_obs, use_image_obs, use_goal, num_streets):
    w = obs.shape[1]
    out = {}
    if not use_image_obs:
        out['image'] = np.zeros((3, w, w))
    else:
        out['image'] = obs[0:3]

    if use_gps_obs:
        out['rel_gps'] = obs[3, 0, :4]

    if use_visible_text_obs:
        out['visible_street_names'] = obs[4, 0:2, :num_streets].reshape((2, num_streets))
        out['visible_house_numbers'] = obs[5, 0:3, :40].reshape((3, 4, 10))

    if use_goal:
        out['goal_house_numbers'] = obs[6, 0, :40].reshape((4, 10))
        out['goal_street_names'] = obs[7, 0, :num_streets]

    return out
