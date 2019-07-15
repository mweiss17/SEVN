import numpy as np

def norm_angle(x):
    # Utility function to keep some angles in the space of -180 to 180 degrees
    if x > 180:
        x = -360 + x
    elif x < -180:
        x = 360 + x
    return x

def convert_house_numbers(num):
    res = np.zeros((4, 10))
    for col, row in enumerate(str(num)):
        res[col, int(row)] = 1
    return res.reshape(-1)

def convert_house_vec_to_ints(vec):
    numbers = []
    for offset in range(0, 120, 10):
        numbers.append(str(vec[offset:offset + 10].argmax()))
    return (int("".join(numbers[:4])), int("".join(numbers[4:8])), int("".join(numbers[8:12])))

def convert_street_name(street_name, all_street_names):
    assert street_name in all_street_names
    return (all_street_names == street_name).astype(int)

def normalize_image(image):
    # Values calculated for SEVN-mini: mean=[0.437, 0.452, 0.479], std=[0.2495, 0.2556, 0.2783]
    normed_image = image / 255.0
    normed_image[:, :, 0] = (normed_image[:, :, 0] - 0.437) / 0.2495
    normed_image[:, :, 1] = (normed_image[:, :, 1] - 0.452) / 0.2556
    normed_image[:, :, 2] = (normed_image[:, :, 2] - 0.479) / 0.2783
    return normed_image
