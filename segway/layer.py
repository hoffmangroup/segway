#!/usr/bin/env python
from __future__ import division, with_statement

"""
layer: convert flattened Viterbi BED files to a layered thick/thin bed file
"""

__version__ = "$Revision$"

# Copyright 2009-2011 Michael M. Hoffman <mmh1@washington.edu>

from collections import defaultdict
import re
import sys

from numpy import array, diff, vstack
from tabdelim import DictReader

from .bed import get_trackline_and_reader_native
from .task import BED_SCORE, BED_STRAND
from ._util import get_label_color, maybe_gzip_open

BED_START = "0"
ACCEPTABLE_STRANDS = set(".+")

OFFSET_START = 0
OFFSET_END = 1
OFFSET_LABEL = 2

TRACKLINE_DEFAULT = ["track", 'description="segway-layer output"',
                     "visibility=full"]

class PassThroughDict(dict):
    def __missing__(self, key):
        return key

# XXX: don't need counter, can use length
class IncrementingDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        defaultdict.__init__(self, *args, **kwargs)
        self.counter = 0

    def __missing__(self, key):
        self[key] = value = self.counter
        self.counter += 1

        return value

def make_comment_ignoring_dictreader(iterable, *args, **kwargs):
    return DictReader((item for item in iterable if not item.startswith("#")),
                      *args, **kwargs)

def load_mnemonics(filename):
    mnemonics = PassThroughDict()
    ordering = []

    if filename is not None:
        with open(filename) as infile:
            for row in make_comment_ignoring_dictreader(infile):
                mnemonics[row["old"]] = row["new"]
                ordering.append(row["old"])

    ordering.reverse()

    return mnemonics, ordering

def uniquify(seq):
    """
    http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order

    http://www.peterbe.com/plog/uniqifiers-benchmark
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if x not in seen and not seen_add(x)]

re_stem = re.compile(r"^(.+(?=\.)|[A-Za-z]+|)")
def get_stem(text):
    # returns empty string when there is no stem part
    return re_stem.match(text).group(0)

def recolor(mnemonics, labels):
    res = {}

    stem_colors = IncrementingDefaultDict()

    for label in labels:
        stem = get_stem(mnemonics[label])

        res[label] = get_label_color(stem_colors[stem])

    return res

# XXX: it would be better to not define the coordinates in non-contiguous areas
def layer(infilename="-", outfilename="-", mnemonic_filename=None,
          trackline_updates={}):
    # dict of lists of tuples of ints
    segments_dict = defaultdict(list)
    colors = {} # overwritten each time
    label_dict = IncrementingDefaultDict()

    mnemonics, ordering = load_mnemonics(mnemonic_filename)

    with maybe_gzip_open(infilename) as infile:
        trackline, reader = get_trackline_and_reader_native(infile)
        if not trackline:
            trackline = TRACKLINE_DEFAULT[:]

        for datum in reader:
            try:
                assert datum.strand in ACCEPTABLE_STRANDS
            except AttributeError:
                pass

            label = datum.name

            # XXX: somewhat duplicative of segtools.__init__.Annotations
            label_key = label_dict[label]

            segment = (datum.chromStart, datum.chromEnd, label_key)
            segments_dict[datum.chrom].append(segment)

            try:
                colors[label] = datum.itemRgb
            except AttributeError:
                try:
                    label_int = int(label)
                except ValueError:
                    label_int = label_key

                colors[label] = get_label_color(label_int)

    for word_index, word in enumerate(trackline):
        if word == "visibility=dense":
            trackline[word_index] = "visibility=full"

    # XXX: inefficient. Python 2.7+'s collections.OrderedDict would be better
    for key, value in trackline_updates.iteritems():
        start = key + "="
        for word_index, word in enumerate(trackline):
            if word.startswith(start):
                trackline[word_index] = '%s="%s"' % (key, value)
                break

    labels_sorted = uniquify(ordering + sorted(colors.iterkeys()))

    # used to only do this sometimes, don't know why:
    # if len(mnemonics) == len(colors):
    colors = recolor(mnemonics, labels_sorted)

    with maybe_gzip_open(outfilename, "w") as outfile:
        print >>outfile, " ".join(trackline)

        for chrom, segments in segments_dict.iteritems():
            segments_array = array(segments)

            end = segments_array.max()

            for label in labels_sorted:
                label_key = label_dict[label]
                color = colors[label]

                segments_label_key_rows = segments_array[:, OFFSET_LABEL] == label_key
                segments_label_key = segments_array[segments_label_key_rows,
                                                OFFSET_START:OFFSET_END+1]

                # XXX: this will probably break when there is a zero
                # in the data
                segments_label_key_augmented = vstack([(0, 0), segments_label_key,
                                                   (end-1, end)])

                block_count = str(len(segments_label_key_augmented))

                # XXX: repetitive _str
                block_sizes = diff(segments_label_key_augmented).ravel()
                block_sizes_str = ",".join(map(str, block_sizes))

                block_starts = segments_label_key_augmented[:, 0]
                block_starts_str = ",".join(map(str, block_starts))

                # just passes through the label itself if there are no mnemonics
                mnemonic = mnemonics[str(label)]

                row = [chrom, BED_START, str(end), mnemonic, BED_SCORE,
                       BED_STRAND, BED_START, str(end), color, block_count,
                       block_sizes_str, block_starts_str]

                print >>outfile, "\t".join(row)

def parse_options(args):
    from optplus import OptionParser

    usage = "%prog [OPTION]... [INFILE] [OUTFILE]"
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)
    parser.add_option("-m", "--mnemonic-file", metavar="FILE",
                      help="specify tab-delimited file with mnemonic "
                      "replacement identifiers for segment labels")
    parser.add_option("-s", "--track-line-set", metavar="ATTR VALUE",
                      action="update", help="set ATTR to VALUE in track line",
                      default={})

    options, args = parser.parse_args(args)

    if not len(args) <= 2:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return layer(mnemonic_filename=options.mnemonic_file,
                 trackline_updates=options.track_line_set, *args)

if __name__ == "__main__":
    sys.exit(main())
