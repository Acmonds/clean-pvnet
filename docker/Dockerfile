FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel


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

# conda
ENV PATH $PATH:/opt/conda/envs/env/bin:/opt/conda/bin

# Clone the repository and install Python requirements
RUN cd /home && \
    git clone https://github.com/Acmonds/clean-pvnet clean-pvnet && \
    cd clean-pvnet && \
    pip install --user Cython==0.28.2 && \
    pip install -r requirements.txt && \
    pip install --user transforms3d

# Setup custom C++/CUDA extensions
RUN cd /home/clean-pvnet/lib/csrc && \
    export CUDA_HOME=/usr/local/cuda && \
    cd ransac_voting && \
    python setup.py build_ext --inplace && \
    cd ../nn && \
    python setup.py build_ext --inplace && \
    cd ../fps && \
    python setup.py build_ext --inplace

WORKDIR /home/clean-pvnet

CMD ["/bin/bash"]
