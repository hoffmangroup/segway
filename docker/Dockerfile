FROM ubuntu:latest

# Install Ubuntu 20.04.01 LTS
# Install pkg-config for locationg development files (hdf5)
# Install Python 3 and pip3
# Install HDF5
# Install Git
# Install curl (for retrieving GMTK)
# Avoid prompts and do not install recommended packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    python3 \
    python3-pip \
    libhdf5-dev \
    hdf5-tools \
    git \
    curl \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# Install GMTK
# Downloads location: http://melodi.ee.washington.edu/cgi-bin/gmtk-download.pl
# Ubuntu 20.04 defaults to a more recent c++ version; need to specify c++11 with CXXFLAGS
# TODO: Check to see if the image would be smaller if using ADD of the URL,
# skipping install of curl and then using RUN for the remaining gmtk commands
# Previous download location: http://melodi.ee.washington.edu/downloads/gmtk/gmtk-1.4.4.tar.gz 
# New (temporary?) location: https://depot.galaxyproject.org/software/gmtk/gmtk_1.4.4_src_all.tar.gz
RUN curl -SL https://depot.galaxyproject.org/software/gmtk/gmtk_1.4.4_src_all.tar.gz \
    | tar -xz \
    && cd gmtk-1.4.4 \
    && ./configure CXXFLAGS='-std=c++11'\
    && make \
    && make install \
    && cd .. \
    && rm -rf gmtk-1.4.4

# Update pip
# Install PyFlakes for testing
RUN python3 -m pip install --upgrade \
    pip \
    setuptools \
    pyflakes

# Install segway
# RUN pip install segway
