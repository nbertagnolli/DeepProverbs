# Container for running DeepProverbs models on the CPU
# This Docker container does not support GPU the purpose
# is to allow people to quickly play with the model and code

FROM ubuntu:16.04

# Install libraries necessary for python and Tensorflow
RUN apt-get update && apt-get install -y \
    curl \
    libfreetype6-dev \
    libpng12-dev \
    libzmq3-dev \
    pkg-config \
    git \
    libhdf5-dev \
    graphviz \
    libenchant1c2a

# Install Python and necessary libraries
RUN apt-get -y update && apt-get -y install \
    python \
    python-pip \
    python-dev

# Install Python Libraries
RUN pip install --upgrade pip && \
    pip install numpy && \
    pip install pandas && \
    pip install matplotlib && \
    pip install h5py && \
    pip install pydot-ng && \
    pip install graphviz && \
    pip install tensorflow && \
    pip install keras && \
    pip install pyenchant && \
    pip install tweepy

CMD /bin/bash

