import os
from shutil import copyfile
os.chdir("crops")
for fname in os.listdir():
  s_fname = fname.split("_")
  if fname == ".DS_Store":
    continue
  print(fname)
  if int(s_fname[3].split(".")[0]) % 10 == 0 and fname.startswith("camera_2"):
    copyfile(fname, "../10x_reduction/" + fname)
