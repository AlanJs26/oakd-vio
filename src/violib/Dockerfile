FROM ubuntu:jammy

RUN apt-get clean && apt-get update
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  apt-utils \
  build-essential \
  gcc \
  g++ \
  clang \
  cmake \
  git \
  python3-dev \
  python3-numpy \
  libopencv-dev

# Create dev user with password 'dev'
RUN useradd -ms /bin/bash dev
RUN echo 'dev:dev' | chpasswd

# RUN mkdir -p /home/dev/soft-slam

COPY . /home/dev/soft-slam

WORKDIR /home/dev/soft-slam

RUN mkdir build &&\
    cd build &&\
    cmake .. &&\
    make -j4
