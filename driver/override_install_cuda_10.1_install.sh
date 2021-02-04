#!/bin/bash
cd /media/b/TOSHIBA\ EXT/2_cuda_cudnn_anaconda/CUDA_10.1_RELATED/cuda10.0
echo 'First input "accept", then press enter'
echo 'Select CUDA toolkit only'
sudo sh cuda_10.1.168_418.67_linux.run --override
echo "export"" ""PATH=\"/usr/local/cuda-10.1/bin:\$PATH\"" >> ~/.bashrc
echo "export"" ""LD_LIBRARY_PATH=\"/usr/local/cuda-10.1/lib64:\$LD_LIBRARY_PATH\"" >> ~/.bashrc

cat /usr/local/cuda/version.txt


