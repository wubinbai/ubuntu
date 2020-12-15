#!/bin/bash

python3 -m pip install -U setuptools pip


sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install g++-6
export NVCC="nvcc --compiler-bindir gcc-6"

#v9.0
#
#
#$ pip install cupy-cuda90
#
#v9.2
#
#
#$ pip install cupy-cuda92
#
#v10.0
#
#
#$ pip install cupy-cuda100
#
#v10.1
#
#
#$ pip install cupy-cuda101
#
#v10.2
#
#
##$ pip install cupy-cuda102
#
#v11.0
#
#
#$ pip install cupy-cuda110
#
#v11.1
#
#
#$ pip install cupy-cuda111
#
#
