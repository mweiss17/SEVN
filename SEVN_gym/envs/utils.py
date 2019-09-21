import math
import numpy as np
import pandas as pd

def norm_angle(x):
    '''
    Utility function to keep some angles in the space of -180 to 180 degrees.
    '''
    if x > 180:
        x = -360 + x
    elif x < -180:
        x = 360 + x
    return x


def norm_angle_360(x):
    '''
    Utility function to go from -180/180 degrees to 0/360.
    '''
    if x < 0:
        x = 360 + x
    return x


def get_angle_between_nodes(G, n1, n2):
    '''
    Calculates the angle between two nodes
    '''
    x = G.nodes[n2]['coords'][0] - G.nodes[n1]['coords'][0]
    y = G.nodes[n2]['coords'][1] - G.nodes[n1]['coords'][1]
    angle = (math.atan2(y, x) * 180 / np.pi)
    return angle


def smallest_angle(a1, a2):
    '''
    Utility function to find smallest angle between two angles in the space of
    -180 to 180 degrees.
    '''
    angle = a2 - a1
    return (angle + 180) % 360 - 180


def smallest_angles(a1, a2):
    angles = []
    for a in a2:
        angle = a - a1
        angles.append(np.abs((angle + 180) % 360 - 180))
    return angles


def normalize_image(image):
    '''
    Values calculated for SEVN-mini: mean=[0.437, 0.452, 0.479],
    std=[0.2495, 0.2556, 0.2783].
    '''
    normed_image = image / 255.0
    normed_image[:, :, 0] = (normed_image[:, :, 0] - 0.437) / 0.2495
    normed_image[:, :, 1] = (normed_image[:, :, 1] - 0.452) / 0.2556
    normed_image[:, :, 2] = (normed_image[:, :, 2] - 0.479) / 0.2783
    return normed_image


def denormalize_image(normed_image):
    '''
    Values calculated for SEVN-mini: mean=[0.437, 0.452, 0.479],
    std=[0.2495, 0.2556, 0.2783].
    '''
    normed_image[:, :, 0] = (normed_image[:, :, 0] * 0.2495) + 0.437
    normed_image[:, :, 1] = (normed_image[:, :, 1] * 0.2556) + 0.452
    normed_image[:, :, 2] = (normed_image[:, :, 2] * 0.2783) + 0.479
    return normed_image


def convert_house_numbers(num):
    res = np.zeros((4, 10))
    for col, n in enumerate(str(num).zfill(4)):
        n = int(n)
        if n > 0:
            res[col, n] = 1
    return res.reshape(-1)


def convert_house_vec_to_ints(vec):
    numbers = []
    hn_len = 40
    for i in range(int(vec.size/hn_len)):
        number = []
        for offset in range(i*hn_len, i*hn_len + hn_len, 10):
            number.append(str(vec[offset:offset + 10].argmax()))
        numbers.append(int("".join(number)))
    return numbers


def convert_street_name(street_name, all_street_names):
    assert street_name in all_street_names
    return (all_street_names == street_name).astype(int)


def sample_gps(groundtruth, x_scale, y_scale, noise_scale=0):
    coords = groundtruth[['x', 'y']]
    x = (coords.x + np.random.normal(loc=0.0, scale=noise_scale)) / x_scale
    y = (coords.y + np.random.normal(loc=0.0, scale=noise_scale)) / y_scale
    return (x, y)


def extract_text(x, w, subset, high_res):
    house_numbers = []
    street_signs = []

    if isinstance(subset, pd.Series):
        if high_res:
            x_min = subset.x_min * 3840 / 224
            x_max = subset.x_max * 3840 / 224
        else:
            x_min = subset.x_min
            x_max = subset.x_max
        if x < x_min and x + w > x_max:
            if subset.obj_type == 'house_number':
                house_numbers.append(subset.house_number)
            elif subset.obj_type == 'street_sign':
                street_signs.append(subset.street_name)
        return house_numbers, street_signs

    for _, row in subset.iterrows():
        if high_res:
            x_min = row.x_min * 3840 / 224
            x_max = row.x_max * 3840 / 224
        else:
            x_min = row.x_min
            x_max = row.x_max

        if x < x_min and x + w > x_max:
            if row.obj_type == 'house_number':
                house_numbers.append(row.house_number)
            elif row.obj_type == 'street_sign':
                street_signs.append(row.street_name)
    return house_numbers, street_signs


def stack_text(house_numbers, street_signs, num_streets):
    visible_text = {}
    temp = np.zeros(120)
    if len(house_numbers) != 0:
        nums = np.hstack(house_numbers)[:120]
        temp[:nums.size] = nums
    visible_text['house_numbers'] = temp

    vec_size = 2 * num_streets
    temp = np.zeros(vec_size)
    if len(street_signs) != 0:
        nums = np.hstack(street_signs)[:vec_size]
        temp[:nums.size] = nums
    visible_text['street_names'] = temp
    return visible_text
