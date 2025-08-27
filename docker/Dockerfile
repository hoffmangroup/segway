FROM ubuntu:latest

# Install Latest Ubuntu LTS
# Install pkg-config for GMTK
# Install curl (for retrieving GMTK)
# Install HDF5 (for genomedata dependency)
# Install git for github actions checkout to function properly
# Avoid prompts and do not install recommended packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libhdf5-serial-dev \
    hdf5-tools \
    ca-certificates \
    curl \
    git \
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

# Let CI machines manage Python and segway installation
# RUN pip install segway
