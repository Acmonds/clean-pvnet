FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

# 设置时区并安装基本依赖
RUN ln -fs /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime && \
    apt-get update && apt-get install -y \
    nano \
    wget \
    cmake \
    git \
    ca-certificates \
    python3-pyqt5 \
    libjpeg-dev \
    zip \
    unzip \
    libpng-dev \
    libeigen3-dev \
    libglfw3-dev \
    libglfw3 \
    libgoogle-glog-dev \
    libsuitesparse-dev \
    libatlas-base-dev \
    libssl-dev \
    zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 由于您使用的是PyTorch的官方镜像，该镜像中已经包含了conda环境，所以无需重新安装miniconda
ENV PATH $PATH:/opt/conda/envs/env/bin:/opt/conda/bin

RUN cd /home && \
    git clone https://github.com/zju3dv/clean-pvnet.git pvnet && \
    cd pvnet && \
    pip install --user Cython==0.28.2 && \
    pip install -r my_requirements.txt && \
    pip install --user transforms3d && \
    cd .. && \
    rm -rf pvnet

CMD ["/bin/bash"]
