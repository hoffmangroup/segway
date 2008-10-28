#!/usr/bin/env python
from __future__ import division, with_statement

"""
h5histogram: prints histogram
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

import sys

from functools import partial
from numpy import array, histogram, isfinite, NINF, PINF
from tables import openFile

from ._util import get_col_index, walk_continuous_supercontigs

BINS = 100
FIELDNAMES = ["lower_edge", "count"]

def calc_range(trackname, filenames):
    minimum = PINF
    maximum = NINF

    for filename in filenames:
        with openFile(filename) as chromosome:
            col_index = get_col_index(trackname)

            attrs = chromosome.root._v_attrs
            minimum = min(minimum, attrs.mins[col_index])
            maximum = max(maximum, attrs.maxs[col_index])

    return minimum, maximum

def calc_histogram(trackname, filenames, data_range):
    histogram_custom = partial(histogram, bins=BINS, range=data_range,
                               new=True)

    hist, edges = histogram_custom(array([]))

    for filename in filenames:
        with openFile(filename) as chromosome:
            print >>sys.stderr, filename

            col_index = get_col_index(trackname)

            supercontig_walker = walk_continuous_supercontigs(chromosome)
            for supercontig, continuous in supercontig_walker:
                col = continuous[:, col_index]
                col_finite = col[isfinite(col)]
                hist_supercontig, edges_supercontig = \
                    histogram_custom(col_finite)

                assert edges_supercontig == edges
                hist += hist_supercontig

    return hist, edges

def print_histogram(hist, edges, maximum):
    for row in zip(edges, hist.tolist() + ["NA"]):
        print "\t".join(map(str, row))

def h5histogram(trackname, filenames):
    print "\t".join(FIELDNAMES)

    # two passes to avoid running out of memory
    data_range = calc_range(trackname, filenames)
    hist, edges = calc_histogram(trackname, filenames)

    print_histogram(hist, edges)

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... FILE..."
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)
    parser.add_option("-c", "--col", type=int, metavar="COL",
                      default=0, help="write values in column COL")

    options, args = parser.parse_args(args)

    if not len(args) >= 1:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return h5histogram(options.col, args)

if __name__ == "__main__":
    sys.exit(main())
