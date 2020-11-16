FROM ufoym/deepo:keras
LABEL maintainer="wubinbai@yahoo.com"
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update

RUN apt-get install -y libsndfile1

RUN pip3 install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com --no-cache-dir llvmlite==0.31.0 numba==0.48.0 librosa==0.7.2


