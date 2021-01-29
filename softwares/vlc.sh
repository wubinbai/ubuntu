#! /bin/bash

# Install VLC Player for playing mp4

sudo add-apt-repository ppa:videolan/master-daily
sudo apt update

# To install:

sudo apt install -y vlc qtwayland5

#In order to use the streaming and transcode features in VLC for Ubuntu 18.04, enter the following command to install the libavcodec-extra packages.

sudo apt install -y libavcodec-extra

