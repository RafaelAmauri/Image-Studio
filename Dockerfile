FROM ubuntu:24.04

WORKDIR /data

COPY requirements.txt /data/requirements.txt

# Install all OS related dependencies, pipx and ensure it is available in the PATH
RUN apt-get update && \
    apt-get install -y bash python3 python3-pip pipx && \
    pipx ensurepath && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.utf8

# Install all dependencies
RUN pipx install image-studio && \
    pipx runpip image-studio install -r requirements.txt

COPY . /data
