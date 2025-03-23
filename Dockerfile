ARG CORES=1

FROM ubuntu:noble

RUN apt update\
    && DEBIAN_FRONTEND=noninteractive apt install -y build-essential cmake curl git unzip fzf libopencv-dev libpcl-dev

RUN git clone --recurse-submodules https://github.com/luxonis/depthai-core /depthai-core\
    && cd /depthai-core\
    && cmake -S. -Bbuild -D'BUILD_SHARED_LIBS=ON' -D'CMAKE_INSTALL_PREFIX=/usr/local'\
    && cmake --build build --parallel ${CORES} --target install
   

RUN mkdir -p /root/dev

COPY . /root/dev

WORKDIR /root/dev

# Build project
RUN cmake -S. -Bbuild && cmake --build build --parallel ${CORES}

# Install neovim
RUN curl -fsSL https://raw.githubusercontent.com/AlanJs26/nvim_config/master/install.sh| bash -s -- --nvim-appimage
