#!/bin/bash

sudo apt install -y git vim speedtest-cli unrar retext xsel tree ffmpeg python3-pip ipython3 libsndfile1 sox pv
# pip3 ipython3
# pip3: matplotlib tqdm librosa 
echo '======================================='
echo '               All done!               '
echo '==============='
echo 'git version'
git --version

echo '==============='
echo 'vim version'
vim --version | grep 'IMproved'

echo '==============='
echo 'speedtest-cli version'
speedtest-cli --version

echo '==============='
echo 'unrar version(which command)'
which unrar

echo '==============='
echo 'retext version(which command)'
which retext

echo '==============='
echo 'xsel version'
xsel --version

echo '==============='
echo 'tree version'
tree --version

echo '==============='
echo 'ffmpeg version:'
ffmpeg -version | grep 'version '

echo '==============='
echo 'pip3 version:'
which pip3

echo '==============='
echo 'ipython3 version'
ipython3 --version

echo '==============='
echo 'sox version'
sox --version

echo '==============='
echo 'pv version'
pv --version | grep 'pv '
