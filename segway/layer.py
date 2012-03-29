#!/usr/bin/env python
from __future__ import division, with_statement

"""
layer: convert flattened Viterbi BED files to a layered thick/thin bed file
"""

__version__ = "$Revision$"

# Copyright 2009-2012 Michael M. Hoffman <mmh1@washington.edu>

from collections import defaultdict
import re
import sys
from tempfile import NamedTemporaryFile

from numpy import array, diff, vstack
from optbuild import OptionBuilder_ShortOptWithEquals
from tabdelim import DictReader

from .bed import get_trackline_and_reader_native
from ._util import (BED_SCORE, BED_STRAND, get_label_color,
                    maybe_gzip_open, PassThroughDict, SUFFIX_BED,
                    SUFFIX_TAB)

BED_START = "0"
ACCEPTABLE_STRANDS = set(".+")

OFFSET_START = 0
OFFSET_END = 1
OFFSET_LABEL = 2

TRACKLINE_DEFAULT = ["track", 'description="segway-layer output"',
                     "visibility=full"]

BEDTOBIGBED_PROG = OptionBuilder_ShortOptWithEquals("bedToBigBed")

# XXX: don't need counter, can use length
class IncrementingDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        defaultdict.__init__(self, *args, **kwargs)
        self.counter = 0

    def __missing__(self, key):
        self[key] = value = self.counter
        self.counter += 1

        return value

class Tee(object):
    def __init__(self, *args):
        self._items = args
        self._exits = []

    ## code borrowed from Python 2.7 contextlib.nested
    def __enter__(self):
        items = self._items
        new_items = []
        exits = []

        for item in items:
            exit = item.__exit__
            enter = item.__enter__
            new_items.append(enter())
            exits.append(exit)

        self._items = new_items
        self._exits = exits

        return self

    def __exit__(self, *exc):
        while self._exits:
            exit = self._exits.pop()
            try:
                if exit(*exc):
                    exc = (None, None, None)
            except:
                exc = sys.exc_info()
        if exc != (None, None, None):
            # Don't rely on sys.exc_info() still containing
            # the right information. Another exception may
            # have been raised and caught by an exit method
            raise exc[0], exc[1], exc[2]

    def write(self, *args, **kwargs):
        for item in self._items:
            item.write(*args, **kwargs)

    def flush(self, *args, **kwargs):
        for item in self._items:
            item.flush(*args, **kwargs)

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

def make_layer_filename(filename):
    """
    exported to run.py

    replace .bed from right side with .layered.bed
    """
    left, _, right = filename.rpartition(".bed")

    return ".layered.bed".join([left, right])

# XXX: don't define coordinates in non-contiguous areas
def layer(infilename="-", outfilename="-", mnemonic_filename=None,
          trackline_updates={}, bigbed_outfilename=None):
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

    for key, value in trackline_updates.iteritems():
        start = key + "="
        for word_index, word in enumerate(trackline):
            if word.startswith(start):
                trackline[word_index] = '%s="%s"' % (key, value)
                break

    labels_sorted = uniquify(ordering + sorted(colors.iterkeys()))

    # only do this if you have mnemonics
    if mnemonics:
        colors = recolor(mnemonics, labels_sorted)

    outfile = maybe_gzip_open(outfilename, "w")

    if bigbed_outfilename:
        temp_file = NamedTemporaryFile(prefix=__package__, suffix=SUFFIX_BED)
        bigbed_infilename = temp_file.name
        outfile = Tee(outfile, temp_file)

    ends = {}
    with outfile as outfile:
        try:
            final_outfile = outfile._items[0]
        except AttributeError:
            final_outfile = outfile
        print >>final_outfile, " ".join(trackline)

        for chrom, segments in segments_dict.iteritems():
            segments_array = array(segments)

            end = segments_array.max()
            ends[chrom] = end

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

                # this just passes through the label itself if there
                # are no mnemonics
                mnemonic = mnemonics[str(label)]

                row = [chrom, BED_START, str(end), mnemonic, BED_SCORE,
                       BED_STRAND, BED_START, str(end), color, block_count,
                       block_sizes_str, block_starts_str]

                print >>outfile, "\t".join(row)

        if bigbed_outfilename:
            outfile.flush()

            # XXX: refactor this into a function in _util
            # XXX: want to make a new wrapper to NamedTemporaryFile
            # that closes after one level of context, deletes after
            # the next
            # with MyNamedTemporaryFile() as temp_file:
            #     with temp_file
            #         print >>temp_file, "blah"
            #     print temp_file.name
            with NamedTemporaryFile(prefix=__package__, suffix=SUFFIX_TAB) as sizes_file:
                for chrom, end in ends.iteritems():
                    print >>sizes_file, "\t".join([chrom, str(end)])

                sizes_file.flush()
                BEDTOBIGBED_PROG(bigbed_infilename, sizes_file.name, bigbed_outfilename)

def parse_options(args):
    from optplus import OptionParser

    usage = "%prog [OPTION]... [INFILE] [OUTFILE]"
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)
    parser.add_option("-b", "--bigBed", metavar="FILE",
                      help="specify bigBed output file")
    parser.add_option("-m", "--mnemonic-file", metavar="FILE",
                      help="specify tab-delimited input file with mnemonic "
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
                 trackline_updates=options.track_line_set,
                 bigbed_outfilename=options.bigBed,
                 *args)

if __name__ == "__main__":
    sys.exit(main())
