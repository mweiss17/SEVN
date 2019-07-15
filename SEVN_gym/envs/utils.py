import numpy as np

def norm_angle(x):
    # Utility function to keep some angles in the space of -180 to 180 degrees
    if x > 180:
        x = -360 + x
    elif x < -180:
        x = 360 + x
    return x

def normalize_image(image):
    # Values calculated for SEVN-mini: mean=[0.437, 0.452, 0.479], std=[0.2495, 0.2556, 0.2783]
    normed_image = image / 255.0
    normed_image[:, :, 0] = (normed_image[:, :, 0] - 0.437) / 0.2495
    normed_image[:, :, 1] = (normed_image[:, :, 1] - 0.452) / 0.2556
    normed_image[:, :, 2] = (normed_image[:, :, 2] - 0.479) / 0.2783
    return normed_image

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

def extract_text(x, w, subset, high_res):
    house_numbers = []
    street_signs = []

    for _, row in subset.iterrows():

        if high_res:
            x_min = row.x_min * 3840 / 224
            x_max = row.x_max * 3840 / 224
        else:
            x_min = row.x_min
            x_max = row.x_max

        if x < x_min and x + w > x_max:
            if row.obj_type == "house_number":
                house_numbers.append(row.house_number)
            elif row.obj_type == "street_sign":
                street_signs.append(row.street_name)
    return house_numbers, street_signs

def stack_text(house_numbers, street_signs, num_streets):
    visible_text = {}
    temp = np.zeros(120)
    if len(house_numbers) != 0:
        nums = np.hstack(house_numbers)[:120]
        temp[:nums.size] = nums
    visible_text["house_numbers"] = temp

    vec_size = 2 * num_streets
    temp = np.zeros(vec_size)
    if len(street_signs) != 0:
        nums = np.hstack(street_signs)[:vec_size]
        temp[:nums.size] = nums
    visible_text["street_names"] = temp
    return visible_text
