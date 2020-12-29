#! /bin/bash
cd ./no_password
bash set_no_password.sh
cd ..
cd bash_all
bash set_bash_all.sh
cd ..
cd update
bash set_update_preference_aliyun.sh
cd ..
sudo apt update
cd ~
mkdir -p trash
mkdir -p git
