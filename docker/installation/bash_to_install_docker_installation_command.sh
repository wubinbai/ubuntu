#!/bin/bash

sudo apt update
sudo apt install -y apt-transport-https
sudo apt install -y ca-certificates curl gnupg-agent
sudo apt install -y software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
docker --help

sudo systemctl start docker
sudo systemctl enable docker

