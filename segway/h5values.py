#!/usr/bin/env python
from __future__ import division, with_statement

"""
h5values: prints finite values in a column of a number of HDF5 chromosome
files
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

import sys

from numpy import isfinite
from tables import openFile

from ._util import walk_continuous_supercontigs

def h5values(col_index, filenames):
    for filename in filenames:
        with openFile(filename) as h5file:
            print >>sys.stderr, filename
            supercontig_walker = walk_continuous_supercontigs(h5file)
            for supercontig, continuous in supercontig_walker:
                col = continuous[:, col_index]
                col_finite = col[isfinite(col)]
                print "\n".join(str(num) for num in col_finite)

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

    return h5values(options.col, args)

if __name__ == "__main__":
    sys.exit(main())
