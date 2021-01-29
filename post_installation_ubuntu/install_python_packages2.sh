#! /bin/bash

# === tensorflow and keras ===
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ tensorflow==2.2.0 --default-timeout=100
# For version, check: https://github.com/tensorflow/addons
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ tensorflow-addons==0.10.0
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ keras==2.3.1

# === image classifiers ===
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ image-classifiers

# === lightgbm ===
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ lightgbm

# === xgboost ===
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ xgboost

# === cupy ===
#pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ cupy

# === torch ===
#pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ torch --default-timeout=100
pip3 install torch -i  https://pypi.mirrors.ustc.edu.cn/simple
pip3 install torchvision -i  https://pypi.mirrors.ustc.edu.cn/simple

python3 test_python_packages.py


