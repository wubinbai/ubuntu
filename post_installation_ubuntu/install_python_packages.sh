#! /bin/bash

# upgrade pip3 first
pip3 install --upgrade pip

pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ numpy
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ pandas
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ scipy
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ tqdm
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ matplotlib
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ seaborn

# === sklearn ===
#pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ Cython
#pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ joblib
#pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ scikit-image
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ sklearn

# === librosa ===
#pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ llvmlite==0.31.0 numba==0.48.0 librosa==0.7.2
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ llvmlite numba librosa

python3 test_python_packages.py

# === set ipython preference ===
cp ipython_preference/import_here.py /home/*/.ipython/profile_default/startup

