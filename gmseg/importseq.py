#!/usr/bin/env python
from __future__ import division, with_statement

"""
importseq: DESCRIPTION
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

from errno import EEXIST
from os import extsep
from re import compile
import sys

from path import path
from tables import Filters, openFile

from ._util import LightIterator

MIN_GAP_LEN = 100000

DNA_LETTERS_UNAMBIG = "ACGTacgt"
DNA_LETTERS_UNAMBIG_SET = set("ACGTacgt")

EXT_H5 = "h5"
FILTERS = Filters(complevel=1)
SUPERCONTIG_NAME_FMT = "supercontig_%s"

def create_supercontig(h5file, index, start, end):
    name = SUPERCONTIG_NAME_FMT % index
    supercontig = h5file.createGroup("/", name)

    attrs = supercontig._v_attrs
    attrs.start = start
    attrs.end = end

# matches either a run of gaps or a run of non-gaps
re_gap_segment = compile(r"(?:[%s]+|[^%s]+)" % (DNA_LETTERS_UNAMBIG,
                                                DNA_LETTERS_UNAMBIG))

def importseq_single(h5file, seq):
    supercontig_index = 0
    offset_start = 0 # XXX: rename to something sensible

    for m_segment in re_gap_segment.finditer(seq):
        if m_segment.group()[0] not in DNA_LETTERS_UNAMBIG_SET:
            segment_start = m_segment.start()

            if segment_end - segment_start >= MIN_GAP_LEN:
                if offset_start != segment_start:
                    create_supercontig(h5file, supercontig_index, offset_start,
                                       segment_start)

                    supercontig_index += 1
                offset_start = segment_end
        else:
            # I don't want to extend the final group
            # to include ambiguous letters
            segment_end = m_segment.end()

    if offset_start != segment_end:
        create_supercontig(h5file, supercontig_index, offset_start,
                           segment_end)

def importseq(filenames, outdirname):
    outdirpath = path(outdirname)
    try:
        outdirpath.makedirs()
    except OSError, err:
        if err.errno != EEXIST:
            raise

    for filename in filenames:
        print >>sys.stderr, filename

        with file(filename) as infile:
            for defline, seq in LightIterator(infile):
                h5filename = outdirpath / extsep.join([defline, EXT_H5])
                h5file = openFile(h5filename, "w", defline, filters=FILTERS)
                with h5file as h5file:
                    importseq_single(h5file, seq)

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

    return importseq(args[:-1], args[-1])

if __name__ == "__main__":
    sys.exit(main())
