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

# === jupyter notebook (nbconvert is automatically installed) ===
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ jupyter
#pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ prompt-toolkit==1.0.15
#pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ jupyter-console==5.2.0
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ --upgrade ipython3