#!/usr/bin/env python
from __future__ import division, with_statement

"""
calc_distance: DESCRIPTION
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

from collections import defaultdict
import sys

from numpy import array, dstack, ix_, unique, where
from tabdelim import DictReader
from tables import openFile

from ._util import fill_array, find_segment_starts

FIELDNAMES = ["chrom", "chromStart", "chromEnd"]

def load_bedfiles(filenames):
    res = defaultdict(list)

    for filename in filenames:
        with open(filename) as bedfile:
            reader = DictReader(bedfile, FIELDNAMES)

            for row in reader:
                res[row["chrom"]].append((int(row["chromStart"]),
                                          int(row["chromEnd"])))

    return dict((key, array(value)) for (key, value) in res.iteritems())

def calc_distance(h5filenames, bedfilenames):
    sbjct_all = load_bedfiles(bedfilenames)

    for h5filename in h5filenames:
        with openFile(h5filename) as h5file:
            root = h5file.root
            attrs = root._v_attrs
            chrom = attrs.chrom

            try:
                sbjct = sbjct_all[chrom]
            except KeyError:
                continue

            supercontig_start = attrs.start

            identify = root.identify

            start_pos, labels = find_segment_starts(identify.read())

            seg_labels = unique(labels)

            for seg_label in seg_labels:
                # XXX: duplicative of run.py:print_segment_summary_stats()
                where_seg, = where(labels == seg_label)

                query = start_pos.take([where_seg, where_seg+1])
                query.shape = tuple(reversed(query.shape))
                query += supercontig_start

                # this is like the cross product, but it is a
                # subtraction instead
                sbjct_ix, query_ix = ix_(sbjct.ravel(), query.ravel())
                distances = sbjct_ix - query_ix

                # repeating block of
                # [[sbjct_start-query_start sbjct_start-query_end]
                #  [sbjct_end-query_start   sbjct_end-query_end]]
                sbjct_start_minus_query_end = distances[::2, 1::2]
                sbjct_end_minus_query_start = distances[1::2, ::2]

                overlap = ((sbjct_start_minus_query_end < 0)
                           & (sbjct_end_minus_query_start > 0))
                which_overlap = overlap.sum(0) > 0

                print >>sys.stderr, ("%s/%s: %s/%s"
                                     % (h5filename, seg_label,
                                        which_overlap.sum(), len(query)))

                # compress number of sbjcts to 1
                closest_distances = abs(distances).min(0)

                # reshape so major axis is number of queries
                closest_distances.shape = (len(query), 2)
                closest_distances[which_overlap] = 0

                # compress minor axis (coordinate start or end)
                closest_distances = closest_distances.min(1)

                for closest_distance in closest_distances:
                    print "\t".join(map(str, [seg_label, closest_distance]))

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... H5FILE..."
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)
    parser.add_option("-b", "--bed", action="append", default=[],
                      metavar="BEDFILE", help="compare with BEDFILE")

    options, args = parser.parse_args(args)

    if not len(args) >= 1:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return calc_distance(args, options.bed)

if __name__ == "__main__":
    sys.exit(main())
