FROM ubuntu:latest

# Install Ubuntu 18.04.01 LTS (Bionic Beaver)
# Install pkg-config for locationg development files (hdf5)
# Install python 2.7 and pip
# Install python 3.6 and pip3
# Install HDF5
# Install Mercurial for hgtools (needed to install for testing)
# Install curl (for retrieving GMTK)
RUN apt-get update && apt-get install -y \
    pkg-config \
    python \
    python3 \
    python-pip \
    python3-pip \
    libhdf5-dev \
    hdf5-tools \
    mercurial \
    curl \
&& rm -rf /var/lib/apt/lists/*
# RUN apt-get install -y libbz2-dev liblzo2-dev libblosc-dev libatlas-dev liblapack-dev

# Install GMTK
# Downloads location: http://melodi.ee.washington.edu/cgi-bin/gmtk-download.pl
# Ubuntu 18.04 defaults to a more recent c++ version; need to specify c++11 with CXXFLAGS
# TODO: Check to see if the image would be smaller if using ADD of the URL,
# skipping install of curl and then using RUN for the remaining gmtk commands
RUN curl -SL http://melodi.ee.washington.edu/downloads/gmtk/gmtk-1.4.4.tar.gz \
    | tar -xz \
    && cd gmtk-1.4.4 \
    && ./configure CXXFLAGS='-std=c++11'\
    && make \
    && make install \
    && cd .. \
    && rm -rf gmtk-1.4.4

# Update pip
# Install PyFlakes for testing
RUN python2 -m pip install --upgrade \ 
    pip \
    setuptools \
    pyflakes
RUN python3 -m pip install --upgrade \
    pip \
    setuptools \
    pyflakes

# Install segway
# RUN pip install segway
