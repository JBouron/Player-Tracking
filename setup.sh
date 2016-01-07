#!/bin/bash

# Installs all the dependencies to run the program.

sudo apt-get update
sudo apt-get install -y build-essential
sudo apt-get install -y libgtk2.0-dev
sudo apt-get install -y pkg-config
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev libjasper-dev
sudo apt-get install libsdl2-dev

sudo apt-get -qq install libopencv-dev build-essential checkinstall cmake pkg-config yasm libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev python-dev python-numpy libtbb-dev libqt4-dev libgtk2.0-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils

# Trying to install libxine-dev, this may not work, but this package 
# does not seem to be necessary.
echo Trying to install libxine-dev ...
sudo apt-get install libxine-dev

echo ###   Installation done                    ###