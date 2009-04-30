#!/usr/bin/env python
from __future__ import division, with_statement

"""
layer: convert flattened Viterbi BED files to a layered thick/thin bed file
"""

__version__ = "$Revision$"

# Copyright 2009 Michael M. Hoffman <mmh1@washington.edu>

from collections import defaultdict
import sys

from numpy import array, diff, vstack

from .bed import get_trackline_and_reader_native
from .task import BED_SCORE, BED_STRAND
from ._util import maybe_gzip_open

BED_START = "0"
ACCEPTABLE_STRANDS = set(".+")

OFFSET_START = 0
OFFSET_END = 1
OFFSET_LABEL = 2

# XXX: it would be better to not define the coordinates in non-contiguous areas
def layer(infilename, outfilename):
    # dict of lists of tuples of ints
    segments_dict = defaultdict(list)
    colors = {} # overwritten each time

    with maybe_gzip_open(infilename) as infile:
        trackline, reader = get_trackline_and_reader_native(infile)
        for datum in reader:
            assert datum.strand in ACCEPTABLE_STRANDS

            # XXX: need to support non-integer names
            label = int(datum.name)

            segment = (datum.chromStart, datum.chromEnd, label)
            segments_dict[datum.chrom].append(segment)

            colors[label] = datum.itemRgb

    for word_index, word in enumerate(trackline):
        if word == "visibility=dense":
            trackline[word_index] = "visibility=full"

    with maybe_gzip_open(outfilename, "w") as outfile:
        print >>outfile, " ".join(trackline)

        for chrom, segments in segments_dict.iteritems():
            segments_array = array(segments)

            end = segments_array.max()

            for label in sorted(colors.iterkeys()):
                color = colors[label]

                segments_label_rows = segments_array[:, OFFSET_LABEL] == label
                segments_label = segments_array[segments_label_rows,
                                                OFFSET_START:OFFSET_END+1]

                # XXX: this will probably break when there is a zero
                # in the data
                segments_label_augmented = vstack([(0, 0), segments_label,
                                                   (end-1, end)])

                block_count = str(len(segments_label_augmented))

                # XXX: repetitive _str
                block_sizes = diff(segments_label_augmented).ravel()
                block_sizes_str = ",".join(map(str, block_sizes))

                block_starts = segments_label_augmented[:, 0]
                block_starts_str = ",".join(map(str, block_starts))

                row = [chrom, BED_START, str(end), str(label), BED_SCORE,
                       BED_STRAND, BED_START, str(end), color, block_count,
                       block_sizes_str, block_starts_str]

                print >>outfile, "\t".join(row)

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... INFILE OUTFILE"
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)

    options, args = parser.parse_args(args)

    if not len(args) == 2:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return layer(*args)

if __name__ == "__main__":
    sys.exit(main())
