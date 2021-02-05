#!/bin/bash

#sudo gedit /etc/modprobe.d/blacklist.conf
## append
#blacklist nouveau
#options nouveau modeset=0

sudo sh -c 'sudo echo "blacklist nouveau" >> /etc/modprobe.d/blacklist.conf'
sudo sh -c 'sudo echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist.conf'

sudo update-initramfs -u

sudo apt update
sudo apt install gcc g++ make -y

mkdir -p ~/temp/
cp /media/*/TOSHIBA\ EXT/2_cuda_cudnn_anaconda/1050Ti_graphics_driver/NVIDIA-Linux-x86_64-460.39.run ~/temp
mv ~/temp/NVIDIA-Linux-x86_64-460.39.run ~/temp/driver.run

# then,
sudo telinit 3 # to open control pallate
# 1 cd ~/temp
# optionally, sudo chmod a+x driver.run
# 2 sh driver.run -no-opengl-files
# optionally, sudo sh driver.run -no-opengl-files
# 3 reboot
# optionally, sudo reboot
