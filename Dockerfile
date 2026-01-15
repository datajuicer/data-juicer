# The data-juicer image includes all open-source contents of data-juicer,
# and it will be installed in editable mode.

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# avoid hanging on interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# add aliyun apt source mirrors for faster download in China
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# install some basic system dependencies
RUN apt-get update && apt-get install -y \
    git curl vim wget aria2 openssh-server gnupg build-essential cmake gfortran \
    ffmpeg libsm6 libxext6 libgl1 libglx-mesa0 libglib2.0-0 libosmesa6-dev \
    freeglut3-dev libglfw3-dev libgles2-mesa-dev vulkan-tools \
    libopenblas-dev liblapack-dev postgresql postgresql-contrib libpq-dev \
    software-properties-common gcc-11 g++-11 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200 && \
    rm -rf /var/lib/apt/lists/*

# install Python 3.11
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-venv python3.11-distutils && \
    # set the default Python
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    # install pip
    curl https://bootstrap.pypa.io/get-pip.py   | python3.11 && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# install uv
RUN pip install --no-cache-dir uv -i https://pypi.tuna.tsinghua.edu.cn/simple

# install java
WORKDIR /opt
RUN wget https://aka.ms/download-jdk/microsoft-jdk-17.0.9-linux-x64.tar.gz   -O jdk.tar.gz \
    && tar -xzf jdk.tar.gz \
    && rm -rf jdk.tar.gz \
    && mv jdk-17.0.9+8 jdk
ENV JAVA_HOME=/opt/jdk
ENV PATH=$JAVA_HOME/bin:$PATH

WORKDIR /data-juicer

# install basic dependencies for Data-Juicer
ENV UV_HTTP_TIMEOUT=600
RUN uv pip install --upgrade --no-cache-dir setuptools==69.5.1 setuptools_scm -i https://pypi.tuna.tsinghua.edu.cn/simple --system \
    && uv pip install --no-cache-dir git+https://github.com/datajuicer/recognize-anything.git -i https://pypi.tuna.tsinghua.edu.cn/simple --system

# copy source code and install
COPY . .
RUN uv pip install --no-cache-dir -v -e .[all] -i https://pypi.tuna.tsinghua.edu.cn/simple --system \
    && python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger');  nltk.download('averaged_perceptron_tagger_eng')"

# 最终入口配置
CMD ["/bin/bash"]
