FROM ubuntu:latest

# Install Ubuntu 16.04.01 LTS (Xenial Xerus)
# Install python 2.7.12 and pip
# Install HDF5
# Install curl (for retrieving GMTK)
RUN apt-get update && apt-get install -y \
    python \
    python-pip \
    libhdf5-serial-dev \
    curl \
&& rm -rf /var/lib/apt/lists/*
# RUN apt-get install -y libbz2-dev liblzo2-dev libblosc-dev libatlas-dev liblapack-dev

# Configure HDF5 include and lib directories
ENV C_INCLUDE_PATH /usr/include/hdf5/serial
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/hdf5/serial
ENV HDF5_DIR /usr/lib/x86_64-linux-gnu/hdf5/serial

# Install GMTK
# Downloads location: http://melodi.ee.washington.edu/cgi-bin/gmtk-download.pl
# TODO: Check to see if the image would be smaller if using ADD of the URL,
# skipping install of curl and then using RUN for the remaining gmtk commands
RUN curl -SL http://melodi.ee.washington.edu/downloads/gmtk/gmtk-1.4.4.tar.gz \
    | tar -xz \
    && cd gmtk-1.4.4 \
    && ./configure \
    && make \
    && make install \
    && cd .. \
    && rm -rf gmtk-1.4.4

# Update pip
# Install mercurial for hgtools (needed to install for testing)
# Install PyFlakes for testing
RUN pip install --upgrade \ 
    pip \
    setuptools \
    mercurial \
    pyflakes

# Install segway
# RUN pip install segway
