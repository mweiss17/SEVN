import os
import pandas as pd
import pickle
import cv2
from SEVN_gym.data import DATA_PATH
from shutil import copyfile


label_df = pd.read_hdf(os.path.join(DATA_PATH, 'label.hdf5'), key='df', mode='r')
addresses = pickle.load(open("SEVN_gym/data/addresses.json", 'rb'))

s1 = set()
street_name_map = {'Rue Jean-Talon': 'jean_talon', 'Rue Jean-Talon O': 'jean_talon', 'Rue Saint-Zotique Est': 'saint_zotique', 'Rue Saint-Zotique E': 'saint_zotique', 'St Zotique East': 'saint_zotique', 'Rue Clark': 'clark', 'Avenue Shamrock': 'shamrock', 'Boul St-Laurent':'saint_laurent', 'Rue St-Laurent': 'saint_laurent', 'Rue Saint-Urbain': 'saint_urbain', 'Rue Dante': 'dante'}

# for address in addresses:
#     s1.add(address['street_name'])

frames = []
names = []

for address in addresses:
    try:
        results = label_df[label_df['house_number'] == str(address['street_number'])]
        results = results[results['street_name'] == street_name_map[address['street_name']]]
    except Exception as e:
        continue
    if len(results):
        for idx, result in results.iterrows():
            frames.append(result.name)
            names.append(address['name'])
print(set(names))
print(f"Num Places: {len(set(names))}")
print(f"Total Images: {len(names)}")
for idx, frame in enumerate(frames):
    filename = f"pano_{str(frame).zfill(6)}.png"
    src_path = "SEVN_gym/data/panos/" + filename
    dst_path = "SEVN_gym/data/places_panos/" + f"pano_{str(frame).zfill(6)}_{names[idx]}.png"
    copyfile(src_path, dst_path)
    # img = cv2.imread(path)