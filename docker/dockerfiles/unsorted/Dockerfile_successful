FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
MAINTAINER Wubin
RUN apt-get update&&\
    apt-get -y install unzip python3 python3-pip python3-dev python3-setuptools ffmpeg cmake python3-setuptools &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --no-cache-dir pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --no-cache-dir scikit-build -i https://pypi.tuna.tsinghua.edu.cn/simple
### solving import setuptools problem:
#RUN apt-get install libssl-dev libffi-dev #python-dev build-essential libxml2-dev libxslt1-dev
#RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir llvmlite==0.31.0
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir numba==0.48.0
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir librosa==0.7.2
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir scipy
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir tqdm
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir matplotlib==3.0.3
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir scikit-learn

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir \
    tensorflow-gpu==1.11.0 \
    keras==2.1.5 \
    imgaug==0.2.6

WORKDIR /
COPY ./code .
COPY ./data .
WORKDIR /code
CMD ["python","main.py","--model_name","model1","--train_data_dir","../data/train/"]
