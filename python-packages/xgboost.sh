#!/bin/bash
sudo apt remove --purge cmake
hash -r
sudo snap install cmake --classic
cmake --version
pip3 install xgboost -i https://pypi.tuna.tsinghua.edu.cn/simple

