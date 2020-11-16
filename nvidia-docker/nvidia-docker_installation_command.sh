#1 Setting up NVIDIA Container Toolkit
## official, fanqiang! NEED to climb the Great Wall of China.
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
#### if the 1st curl fails, try:
#curl -sSL https://get.docker.io/gpg | sudo  apt-key add -
#distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
#curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list


# 2 Install the nvidia-docker2 package (and dependencies) after updating the package listing:

sudo apt-get update
sudo apt-get install -y nvidia-docker2

