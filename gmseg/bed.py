#!/usr/bin/env python
from __future__ import division

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@u.washington.edu>

import sys

FIELDNAMES = ["chrom", "chromStart", "chromEnd", # required
              "name", "score", "strand", "thickStart", "thickEnd", "itemRgb",
              "blockCount", "blockSizes", "blockStarts"]

class Datum(object):
    def __init__(self, words):
        self.__dict__ = dict(zip(FIELDNAMES, words))

def read(iterator):
    for line in iterator:
        words = line.split()
        if words[0] == "track":
            raise NotImplementedError

        assert len(words) >= 3

        yield Datum(words)

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
