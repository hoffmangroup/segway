#!/usr/bin/env python
from __future__ import division, with_statement

"""
gtf2bed: DESCRIPTION
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

import sys

from tabdelim import DictReader, DictWriter

# http://genome.ucsc.edu/FAQ/FAQformat
INFIELDNAMES = ["seqname", "source", "feature", "start", "end", "score",
                "strand", "frame", "attributes"]

OUTFIELDNAMES = ["chrom", "chromStart", "chromEnd"]

def gtf2bed(filenames, source=None, feature=None):
    writer = DictWriter(sys.stdout, OUTFIELDNAMES,
                        extrasaction="ignore", header=False)

    if not filenames:
        filenames = ["-"]

    for filename in filenames:
        if filename == "-":
            infile = sys.stdin
        else:
            infile = open(filename)

        with infile:
            reader = DictReader(infile, INFIELDNAMES)

            for row in reader:
                if source is not None:
                    if row["source"] != source:
                        continue

                if feature is not None:
                    if row["feature"] != feature:
                        continue

                chrom = "chr" + row["seqname"]
                chromStart = str(int(row["start"]) - 1)
                chromEnd = row["end"]

                writer.writerow(locals())

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... [FILE...]"
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)
    parser.add_option("-s", "--source", metavar="SOURCE",
                      help="limit to source SOURCE")
    parser.add_option("-f", "--feature", metavar="FEATURE",
                      help="limit to feature FEATURE")

    options, args = parser.parse_args(args)

    if not len(args) >= 0:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return gtf2bed(args, options.source, options.feature)

if __name__ == "__main__":
    sys.exit(main())
