# These run at about 5 fps when writing to my 5400 rpm external disk.
# The camera captures at 30 fps, and in this 4:12 video there are exactly 7576 frames
# this means it runs at 0.188x realtime, so it takes about 20 minutes to run
# There are two tracks though, so it's twice that long.
# That means there are 40 minutes of processing on a macbook pro writing to disk,
# with a final size on disk of 172 gigabytes

# We could reduce the size of the images, reduce the frame rate,
# increase the I/O bandwidth, or write to SSD to speed things up.

ffmpeg -i HET_0034.MP4 -map 0:0 pngs/track_1_output_%06d.png
ffmpeg -i HET_0034.MP4 -map 0:1 pngs/track_2_output_%06d.png
