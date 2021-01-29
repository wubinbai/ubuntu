#! /bin/bash

# upgrade pip3 first
pip3 install --upgrade pip  -i https://pypi.tuna.tsinghua.edu.cn/simple/

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

# == gpustat ===
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ gpustat

# === librosa ===
#pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ llvmlite==0.31.0 numba==0.48.0 librosa==0.7.2
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ llvmlite numba librosa

# === jupyter notebook (nbconvert is automatically installed) ===
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ jupyter
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ prompt-toolkit==1.0.15
#pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ jupyter-console==5.2.0
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ --upgrade ipython3
#pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ --upgrade prompt-toolkit

# === set ipython preference ===
cp ipython_preference/import_here.py /home/*/.ipython/profile_default/startup

# === cv2 ===
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ opencv-python

# === imgaug ===
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ imgaug
