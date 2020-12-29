#! /bin/bash

# === tensorflow and keras ===
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ tensorflow==2.2.0
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ keras==2.3.1

python3 test_python_packages.py

# === set ipython preference ===
cp ipython_preference/import_here.py /home/*/.ipython/profile_default/startup

