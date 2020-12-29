#!/bin/bash
cd /media/b/TOSHIBA\ EXT/2_cuda_cudnn_anaconda/CUDA_10.1_RELATED/cudnn
tar -xzvf cudnn-10.1-linux-x64-v7.6.5.32.tgz
cd cuda
sudo cp ./include/cudnn.h /usr/local/cuda/include
sudo cp ./lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+x /usr/local/cuda/include/cudnn.h

# view version
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
cd ..
rm -r cuda


