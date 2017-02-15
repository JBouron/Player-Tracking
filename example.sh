# This commands runs the software on the video ace_0 in the samples/floorball2/ folder.
# It starts at the frame 300 and ends at the frame 400.
# Using -t 4, the software will run on 4 threads.
# Using -j 10, we will compute every 10 frames.

# The result is saved as result.avi.

./Bachelor_Project ./samples/floorball2/ 0 -s 300 -e 800 -t 4 -j 10