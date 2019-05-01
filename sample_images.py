import os
from shutil import copyfile

for fname in os.listdir("data/crops"):
  s_fname = fname.split("_")
  print(fname)
  if int(s_fname[3].split(".")[0]) % 10 == 0 and fname.startswith("camera_2"):
    copyfile(fname, "data/10x_reduction/" + fname)
