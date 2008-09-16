#!/usr/bin/env python
from __future__ import division, with_statement

"""
importseq: DESCRIPTION
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

from os import extsep
from re import compile
import sys

from ._util import LightIterator

MIN_GAP_LEN = 100000

DNA_LETTERS_UNAMBIG = "ACGTacgt"
DNA_LETTERS_UNAMBIG_SET = set("ACGTacgt")

# matches either a run of gaps or a run of non-gaps
re_gap_segment = compile(r"(?:[%s]+|[^%s]+)" % (DNA_LETTERS_UNAMBIG,
                                                DNA_LETTERS_UNAMBIG))
def importseq(*filenames):
    for filename in filenames:
        with file(filename) as infile:
            for defline, seq in LightIterator(infile):
                supercontig_index = 0

                for m_segment in re_gap_segment.finditer(seq):
                    offset_start = m_segment.start()
                    offset_end = m_segment.end()

                    if m_segment.group()[0] in DNA_LETTERS_UNAMBIG_SET:
                        print "\t".join(map(str, [defline, supercontig_index,
                                                  offset_start, offset_end]))
                        supercontig_index += 1

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... FILE..."
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)

    options, args = parser.parse_args(args)

    if not len(args) >= 1:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return importseq(*args)

if __name__ == "__main__":
    sys.exit(main())
