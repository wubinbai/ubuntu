#! /bin/bash
cd /home/*/trash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
cd -

# maybe be necessary:
#sudo apt-get install -f
