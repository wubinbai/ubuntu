#!/bin/bash
#
# b@20210219
#
mkdir -p ~/trash
cd ~/trash
git clone https://github.com/bilibili/vim-vide
cd vim-vide
./deploy.sh
