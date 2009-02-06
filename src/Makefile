LDFLAGS+=-lhdf5
CFLAGS=-std=c99 -march=nocona -pipe -fomit-frame-pointer -pedantic -Wall -g -O3

# XXX: use -ggdb3 for debug

all: _load_data
_load_data.o: _load_data.c
_load_data: _load_data.o 
