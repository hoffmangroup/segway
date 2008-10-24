#!/usr/bin/env python
from __future__ import division, with_statement

"""
name_tracks: recursively set up HDF5 files to have named tracks
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

import sys

from numpy import array
from path import path
from tables import openFile

def name_tracks(dirname, *tracknames):
    dirpath = path(dirname)

    for filepath in dirpath.walkfiles():
        with openFile(filepath, "r+") as h5file:
            attrs = h5file.root._v_attrs

            if "tracknames" in attrs:
                raise ValueError("%s already has named tracks" % filepath)

            attrs.tracknames = array(tracknames)

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... DIR TRACKNAME..."
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)

    options, args = parser.parse_args(args)

    if not len(args) >= 2:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return name_tracks(*args)

if __name__ == "__main__":
    sys.exit(main())
