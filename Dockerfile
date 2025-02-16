FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# 安装基础工具和构建依赖
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    cmake \
    ninja-build \
    curl \
    git \
    vim \
    libssl-dev \
    zlib1g-dev \
    pkg-config \
    libboost-all-dev \    
    libfftw3-dev \  
    libopencv-dev \   
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/onnxruntime && \
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.1/onnxruntime-linux-x64-1.16.1.tgz -O /tmp/onnxruntime.tgz && \
    tar -xzf /tmp/onnxruntime.tgz -C /opt/onnxruntime --strip-components=1 && \
    rm /tmp/onnxruntime.tgz

WORKDIR /dom

COPY . /dom

CMD ["/bin/bash"]