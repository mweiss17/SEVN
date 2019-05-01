"""Crops and rotates Vuze images"""

# This ran at about 1 fps (that's 8 cropped rotated images)
# ran on hard drive, definitely disk I/O blocked. so there's improvement to be made there.
# N.B. requires python 3.6+ to run because of the print statement :P 
import os
import cv2
os.chdir("pngs")

W = int(3200 / 2)
H = int(2176 / 2)
import pdb; pdb.set_trace()
for fname in os.listdir():
    if fname == ".DS_Store": continue
    img = cv2.imread(fname)
    frame_num = fname.split(".")[0].split("_")[-1]
    track_num = 1 if "track_1" in fname else 2
    i = 0
    print("Processing: " + fname)
    for x, y in [(0, 0), (W, 0), (0, H), (W, H)]:
        i += 1
        camera_num = i if track_num == 1 else 4 + i
        out_fname = f"../crops/camera_{camera_num}_frame_{frame_num}.png"
        if os.path.isfile(out_fname):
            continue

        crop_img = img[y:y+H, x:x+W]
        out_img = cv2.rotate(crop_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # This only works in python 3.6+
        cv2.imwrite(out_fname, out_img)
