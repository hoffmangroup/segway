#!/usr/bin/env python
from __future__ import division, with_statement

"""
load_seq: DESCRIPTION
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

from errno import EEXIST
from os import extsep
from re import compile, VERBOSE
import sys

from numpy import frombuffer
from path import path
from tables import openFile, UInt8Atom

from ._util import EXT_GZ, FILTERS_GZIP, gzip_open, LightIterator

MIN_GAP_LEN = 100000
assert not MIN_GAP_LEN % 2 # must be even for division

# sre_constants.MAXREPEAT is 65535, so I have to break repeats into
# two segments
REGEX_SEGMENT_LEN = MIN_GAP_LEN // 2 # max == MAXREPEAT

DNA_LETTERS_UNAMBIG = "ACGTacgt"

EXT_H5 = "h5"
SUPERCONTIG_NAME_FMT = "supercontig_%s"

ATOM = UInt8Atom()

def create_supercontig(h5file, index, seq, start, end):
    name = SUPERCONTIG_NAME_FMT % index
    supercontig = h5file.createGroup("/", name)

    seq_array = frombuffer(seq)
    h5file.createCArray(supercontig, "seq", ATOM, seq_array.shape)

    supercontig.seq[...] = seq_array

    attrs = supercontig._v_attrs
    attrs.start = start
    attrs.end = end

# XXXopt: the all-regex approach is much slower than the hybrid
# approach (3 min to load chr21, 6 min to load chr1), but it is easier
# to understand the code and get it correct
#
# the previous code (r22) might have worked fine. Consider backing down to
# it at some point.
#
# XXXopt: a numpy implementation might be better
re_gap_segment = compile(r"""
(?:([^%s]{%d}[^%s]{%d,})                                  # group(1): ambig
   |                                                      #  OR
   ((?:(?:[%s]+|^)(?:[^%s]{1,%d}[^%s]{,%d}(?![^%s]))*)+)) # group(2): unambig
""" % (DNA_LETTERS_UNAMBIG, REGEX_SEGMENT_LEN,
       DNA_LETTERS_UNAMBIG, REGEX_SEGMENT_LEN,
       DNA_LETTERS_UNAMBIG, DNA_LETTERS_UNAMBIG, REGEX_SEGMENT_LEN,
       DNA_LETTERS_UNAMBIG, REGEX_SEGMENT_LEN-1,
       DNA_LETTERS_UNAMBIG), VERBOSE)

def read_seq(h5file, seq):
    supercontig_index = 0

    for m_segment in re_gap_segment.finditer(seq):
        seq_unambig = m_segment.group(2)
        if seq_unambig:
            span = m_segment.span()
            create_supercontig(h5file, supercontig_index, seq_unambig, *span)
            supercontig_index += 1
        else:
            assert m_segment.group(1)

def load_seq(filenames, outdirname):
    outdirpath = path(outdirname)
    try:
        outdirpath.makedirs()
    except OSError, err:
        if err.errno != EEXIST:
            raise

    for filename in filenames:
        print >>sys.stderr, filename

        if filename.endswith(EXT_GZ):
            infile = gzip_open(filename)
        else:
            infile = file(filename)

        # needed to process gzip_open stuff
        with infile as infile:
            for defline, seq in LightIterator(infile):
                h5filename = outdirpath / extsep.join([defline, EXT_H5])
                h5file = openFile(h5filename, "w", defline,
                                  filters=FILTERS_GZIP)
                with h5file as h5file:
                    h5file.root._v_attrs.dirty = True
                    read_seq(h5file, seq)

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... FILE... DST"
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)

    options, args = parser.parse_args(args)

    if not len(args) >= 2:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return load_seq(args[:-1], args[-1])

if __name__ == "__main__":
    sys.exit(main())
