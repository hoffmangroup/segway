#!/usr/bin/env python
from __future__ import division

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

import sys

FIELDNAMES = ["chrom", "chromStart", "chromEnd", # required
              "name", "score", "strand", "thickStart", "thickEnd", "itemRgb",
              "blockCount", "blockSizes", "blockStarts"]

class Datum(object):
    def __init__(self, words):
        self.__dict__ = dict(zip(FIELDNAMES, words))
        self._words = tuple(words)

    def __repr__(self):
        return "%s%s" % (self.__class__.__name__, self._words)

class NativeDatum(Datum):
    def __init__(self, *args, **kwargs):
        Datum.__init__(self, *args, **kwargs)

        # zero-based, http://genome.ucsc.edu/FAQ/FAQformat#format1
        self.chromStart = int(self.chromStart)
        self.chromEnd = int(self.chromEnd)

        try:
            self.score = float(self.score)
        except AttributeError:
            self._words = (self.chrom, self.chromStart,
                           self.chromEnd) + self._words[3:]
        else:
            self._words = (self.chrom, self.chromStart, self.chromEnd,
                           self.name, self.score) + self._words[5:]

def read(iterator, datum_cls=Datum):
    for line in iterator:
        words = line.split()
        if words[0] == "track":
            continue # skip

        assert len(words) >= 3

        yield datum_cls(words)

def read_native(*args, **kwargs):
    return read(datum_cls=NativeDatum, *args, **kwargs)

def get_trackline_and_reader(iterator, datum_cls=Datum):
    line = iterator.next()

    assert line.startswith("track")

    return line.split(), read(iterator, datum_cls)

def get_trackline_and_reader_native(*args, **kwargs):
    return get_trackline_and_reader(datum_cls=NativeDatum, *args, **kwargs)

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
