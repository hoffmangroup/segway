#!/usr/bin/env python
from __future__ import absolute_import, division

__version__ = "$Revision$"

# Copyright 2008-2012 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from itertools import chain
import re
import sys

from six.moves import zip

FIELDNAMES = ["chrom", "chromStart", "chromEnd", # required
              "name", "score", "strand", "thickStart", "thickEnd", "itemRgb",
              "blockCount", "blockSizes", "blockStarts"]

class Datum(object):
    def __init__(self, words):
        self.__dict__ = dict(list(zip(FIELDNAMES, words)))
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
        if not words or words[0] == "track":
            continue # skip

        assert len(words) >= 3

        yield datum_cls(words)

def read_native(*args, **kwargs):
    return read(datum_cls=NativeDatum, *args, **kwargs)

def parse_bed4(line):
    """
    alternate fast path
    """
    row = line.split()
    chrom, start, end, seg = row[:4]
    return row, (chrom, start, end, seg)

re_trackline_split = re.compile(r"(?:[^ =]+=([\"'])[^\1]+?\1(?= |$)|[^ ]+)")
def get_trackline_and_reader(iterator, datum_cls=Datum):
    line = next(iterator).rstrip()

    if line.startswith("track"):
        # retrieves group 1 of re_trackline_split match, which is the whole item
        trackline = [match.group(0)
                     for match in re_trackline_split.finditer(line)]
        reader = read(iterator, datum_cls)
    else:
        trackline = []
        reader = read(chain([line], iterator), datum_cls)

    return trackline, reader

def get_trackline_and_reader_native(*args, **kwargs):
    return get_trackline_and_reader(datum_cls=NativeDatum, *args, **kwargs)

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
