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

def read_gtf_write_bed(reader, writer, source, feature, only_5prime):
    for row in reader:
        if source is not None and row["source"] != source:
            continue

        if feature is not None and row["feature"] != feature:
            continue

        chrom = "chr" + row["seqname"]
        chromStart = int(row["start"]) - 1
        chromEnd = int(row["end"])

        if only_5prime:
            strand = row["strand"]

            if strand == "+":
                chromEnd = chromStart + 1
            elif strand == "-":
                chromStart = chromEnd - 1
            else:
                raise ValueError("can't get 5' end of feature without strand")

        writer.writerow(locals())

def gtf2bed(filenames, source=None, feature=None, only_5prime=False):
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

            read_gtf_write_bed(reader, writer, source, feature, only_5prime)

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... [FILE...]"
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)

    # XXX: group: limit
    parser.add_option("-s", "--source", metavar="SOURCE",
                      help="limit to source SOURCE")
    parser.add_option("-f", "--feature", metavar="FEATURE",
                      help="limit to feature FEATURE")

    # XXX: group: options
    parser.add_option("-5", "--only-5prime", action="store_true",
                      help="reduce features to their 5'-most coordinate")

    options, args = parser.parse_args(args)

    if not len(args) >= 0:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return gtf2bed(args, options.source, options.feature, options.only_5prime)

if __name__ == "__main__":
    sys.exit(main())
