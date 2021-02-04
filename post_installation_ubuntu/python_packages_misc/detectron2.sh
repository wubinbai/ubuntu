#!/bin/bash
#1 install ubuntu 20.04 LTS

#2 choose a good source
sudo cp /etc/apt/sources.list /etc/apt/bak.sources.list; sudo cp sources.list_best_20 /etc/apt/sources.list
#3 update
sudo apt update
#sudo ubuntu-drivers autoinstall
#4 install gcc g++ git pip
sudo apt install -y gcc g++ git python3-pip
#5 make dir git
mkdir ~/git
#6 cd to git
cd ~/git
#7 git clone
git clone git://github.com/ninja-build/ninja.git && cd ninja
#8 checkout release
git checkout release
#9 run configure.py
python3 ./configure.py --bootstrap
#10 cp
sudo cp ./ninja /usr/bin
#11 pip install libs
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ opencv-python torch torchvision --default-timeout=100
#12 install detectron2
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git' -i https://pypi.tuna.tsinghua.edu.cn/simple
#13 install cuda --override
#14 install cudnn
