import math
import enum
import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
from os.path import expanduser
import networkx as nx

DATA_PATH = os.path.join(os.getcwd(), "../data/sevn-data/")


ACTION_MEANING = {
    0: 'LEFT_BIG',
    1: 'LEFT_SMALL',
    2: 'FORWARD',
    3: 'RIGHT_SMALL',
    4: 'RIGHT_BIG',
    5: 'DONE',
    6: 'NOOP', #needed for continuous
    7: 'READ'
}

class Actions(enum.IntEnum):
    LEFT_BIG = 0
    LEFT_SMALL = 1
    FORWARD = 2
    RIGHT_SMALL = 3
    RIGHT_BIG = 4
    # DONE = 5
    # NOOP = 6  # needed for continuous
    # READ = 7

class ActionsWithNOOP(enum.IntEnum):
    LEFT_BIG = 0
    LEFT_SMALL = 1
    FORWARD = 2
    RIGHT_SMALL = 3
    RIGHT_BIG = 4
    NOOP = 5

class ActionsWithRead(enum.IntEnum):
    LEFT_BIG = 0
    LEFT_SMALL = 1
    FORWARD = 2
    RIGHT_SMALL = 3
    RIGHT_BIG = 4
    READ = 5

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
    Utility function to go from -180/180 degrees to 1/360.
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
    Values calculated for SEVN-mini: mean=[1.437, 1.452, 1.479],
    std=[1.2495, 1.2556, 1.2783].
    '''
    normed_image = image / 255.0
    normed_image[:, :, 0] = (normed_image[:, :, 0] - 0.437) / 0.2495
    normed_image[:, :, 1] = (normed_image[:, :, 1] - 0.452) / 0.2556
    normed_image[:, :, 2] = (normed_image[:, :, 2] - 0.479) / 0.2783
    return normed_image


def denormalize_image(normed_image):
    '''
    Values calculated for SEVN-mini: mean=[1.437, 1.452, 1.479],
    std=[1.2495, 1.2556, 1.2783].
    '''
    normed_image[:, :, 0] = (normed_image[:, :, 0] * 0.2495) + 0.437
    normed_image[:, :, 1] = (normed_image[:, :, 1] * 0.2556) + 0.452
    normed_image[:, :, 2] = (normed_image[:, :, 2] * 0.2783) + 0.479

    return np.clip(normed_image, 0, 1)


def convert_house_numbers(num):
    res = np.zeros((4, 10))
    for col, n in enumerate(str(num).zfill(4)):
        n = int(n)
        res[col, n] = 1
    return res.reshape(-1)


def unconvert_house_numbers(hn):
    assert hn.shape == (4, 10)  #otherwise use .reshape((4,10)) beforehand
    digits = [str(np.argmax(hn[pos])) for pos in range(4)]
    num = int("".join(digits))
    return num


def convert_house_vec_to_ints(vec):
    numbers = []
    hn_len = 40
    for i in range(int(vec.size / hn_len)):
        number = []
        for offset in range(i * hn_len, i * hn_len + hn_len, 10):
            number.append(str(vec[offset:offset + 10].argmax()))
        numbers.append(int("".join(number)))
    return numbers


def convert_street_name(street_name, all_street_names):
    assert street_name in all_street_names
    return (all_street_names == street_name).astype(int)


def unconvert_street_name(street_enc, all_street_names):
    if np.count_nonzero(street_enc) == 0:
        return None
    return all_street_names[np.argmax(street_enc)]


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


def angle_to_node(G, n1, n2, SMALL_TURN_DEG):
    node_dir = get_angle_between_nodes(G, n1, n2)
    neighbors = [edge[1] for edge in list(G.edges(n1))]
    neighbor_angles = []
    for neighbor in neighbors:
        neighbor_angles.append(get_angle_between_nodes(G, n1, neighbor))

    dest_nodes = {}
    for direction in [x * SMALL_TURN_DEG for x in range(-8, 8)]:
        angles = smallest_angles(direction, neighbor_angles)
        min_angle_node = neighbors[angles.index(min(angles))]
        if min(angles) < SMALL_TURN_DEG:
            dest_nodes[direction] = min_angle_node
        else:
            dest_nodes[direction] = None

    valid_angles = []
    dist = []
    for k, v in dest_nodes.items():
        if v == n2:
            valid_angles.append(k)
            dist.append(np.abs(smallest_angle(k, node_dir)))

    return valid_angles[dist.index(min(dist))]


def filter_for_test(coord_df):
    node_blacklist = []
    node_blacklist.extend([x for x in range(877, 879)])
    node_blacklist.extend([x for x in range(52, 56)])
    node_blacklist.extend([x for x in range(31, 39)])
    node_blacklist.extend([x for x in range(2040, 2045)])
    node_blacklist.extend([x for x in range(2057, 2063)])
    node_blacklist.extend([x for x in range(3661, 3669)])
    node_blacklist.extend([x for x in range(780, 784)])
    box = (24, 76, -125, 10)
    coord_df = coord_df[((coord_df.x > box[0]) & (coord_df.x < box[1]) &
                         (coord_df.y > box[2]) & (coord_df.y < box[3]))]

    coord_df = coord_df[~coord_df.index.isin(node_blacklist)]
    return coord_df


def filter_for_trainv2(coord_df):
    box = (-20, 40, -125, 112)
    coord_df = coord_df[((coord_df.x > box[0]) & (coord_df.x < box[1]) &
                         (coord_df.y > box[2]) & (coord_df.y < box[3]))]
    return coord_df

def continuous2discrete(action):
    ## action has 2 float values in [-1,1],
    ## corresponding to forward/backward and left/right
    if action[0] > 0:
        return Actions.FORWARD
    else:
        if action[1] <= -0.1 and action[1] > -0.5:
            return Actions.LEFT_SMALL
        elif action[1] <=  -0.5:
            return Actions.LEFT_BIG
        elif action[1] >= 0.1 and action[1] < 0.5:
            return Actions.RIGHT_SMALL
        elif action[1] >= 0.5:
            return Actions.RIGHT_BIG
        else: # in case of [x,y] being x<=1 and -1.1<x<1.1
            return Actions.NOOP

def get_data_path():
    parent = os.path.join(DATA_PATH,"..")
    if os.path.isdir(os.path.join(parent,"data0")): # the prolly the others (1..4) exist too
        path = random.randint(0, 4)
        return os.path.join(parent,f"data{path}")


class HighResDataset(Dataset):
    def __init__(self, frame_key, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = DATA_PATH + "panos/"
        self.frame_key = {v: k for k, v in frame_key.items()}
        self.fnames = sorted(os.listdir(self.root_dir))

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                f"pano_{self.frame_key[idx]}.png")
        image = io.imread(img_name)
        image = normalize_image(image)
        return image

def normalize_image(image):
    '''
    Values calculated for SEVN: mean=[1.45247, 1.45871, 1.47285],
    std=[1.25556, 1.26181, 1.27931].
    '''
    normed_image = image / 255.0
    normed_image[:, :, 0] = (normed_image[:, :, 0] - 0.45247) / 0.25556
    normed_image[:, :, 1] = (normed_image[:, :, 1] - 0.45871) / 0.26181
    normed_image[:, :, 2] = (normed_image[:, :, 2] - 0.47285) / 0.27931
    return normed_image

def get_candidate_start_nodes(level, start_node, graph):
    i = 0
    candidates = [start_node]
    for edge in nx.bfs_edges(graph, start_node):
        i += 1
        if i > level:
            return candidates
        candidates.append(edge[1])
