#! /bin/bash
cd ../driver
./both_cuda_10.1_cudnn_7.6.5_install.sh
cd ../post_installation_ubuntu
./install_ubuntu_softwares.sh
./install_python_packages1.sh

