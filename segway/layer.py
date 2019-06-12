#!/usr/bin/env python
from __future__ import absolute_import, division, with_statement, print_function

"""
layer: convert flattened Viterbi BED files to a layered thick/thin bed file
"""

# Copyright 2009-2012 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from collections import defaultdict
import re
import sys
from tempfile import NamedTemporaryFile

from numpy import array, diff, vstack
from optbuild import OptionBuilder_ShortOptWithEquals
from six import iterkeys, reraise, viewitems
from six.moves import map
from tabdelim import DictReader

from .bed import get_trackline_and_reader_native
from ._util import (BED_SCORE, BED_STRAND, get_label_color,
                    maybe_gzip_open, memoized_property, PassThroughDict,
                    SUFFIX_BED, SUFFIX_TAB)

from .version import __version__

BED_START = "0"
ACCEPTABLE_STRANDS = set(".+")

OFFSET_START = 0
OFFSET_END = 1
OFFSET_LABEL = 2

TRACKLINE_DEFAULT = ["track", 'description="segway-layer output"',
                     "visibility=full"]

BEDTOBIGBED_PROG = OptionBuilder_ShortOptWithEquals("bedToBigBed")

COLOR_DEFAULT = "128,128,128"

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
            reraise(exc[0], exc[1], exc[2])

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

def make_csv(seq):
    return ",".join(map(str, seq))

def update_trackline(trackline, updates):
    if not trackline:
        trackline = TRACKLINE_DEFAULT[:]

    for word_index, word in enumerate(trackline):
        if word == "visibility=dense":
            trackline[word_index] = "visibility=full"

    for key, value in viewitems(updates):
        start = key + "="
        for word_index, word in enumerate(trackline):
            if word.startswith(start):
                trackline[word_index] = '%s="%s"' % (key, value)
                break

def get_color(datum, label, label_key):
    try:
        return datum.itemRgb
    except AttributeError:
        # assign color if it's missing
        try:
            label_int = int(label)
        except ValueError:
            label_int = label_key

        return get_label_color(label_int)

class Segmentation(defaultdict):
    """
    defaultdict of chromosomes:
    chromosome: list of runs
    run: list of segments
    segment: 3-tuple of ints
    """
    def __init__(self):
        defaultdict.__init__(self, list)
        self.colors = {}
        self.label_dict = IncrementingDefaultDict()

    def load(self, infilename):
        colors = self.colors
        label_dict = self.label_dict

        with maybe_gzip_open(infilename) as infile:
            self.trackline, reader = get_trackline_and_reader_native(infile)
            for datum in reader:
                try:
                    assert datum.strand in ACCEPTABLE_STRANDS
                except AttributeError:
                    pass # no strand

                label = datum.name
                label_key = label_dict[label]

                start = datum.chromStart
                chromosome = self[datum.chrom]

                try:
                    run = chromosome[-1]
                except IndexError:
                    run = []
                    chromosome.append(run)
                else:
                    if run[-1][OFFSET_END] != start:
                        run = []
                        chromosome.append(run)

                segment = (datum.chromStart, datum.chromEnd, label_key)
                run.append(segment)

                colors[label] = get_color(datum, label, label_key)

    def load_mnemonics(self, mnemonic_filename):
        self.mnemonics, self.ordering = load_mnemonics(mnemonic_filename)

    @memoized_property
    def labels_sorted(self):
        return uniquify(self.ordering + sorted(iterkeys(self.colors)))

    def update_trackline(self, updates):
        update_trackline(self.trackline, updates)

    def recolor(self):
        mnemonics = self.mnemonics

        # only do this if you have mnemonics
        if mnemonics:
            self.colors = recolor(mnemonics, self.labels_sorted)

    def save(self, outfilename, bigbed_outfilename):
        outfile = maybe_gzip_open(outfilename, "wt")

        if bigbed_outfilename:
            temp_file = NamedTemporaryFile(prefix=__package__, suffix=SUFFIX_BED, mode="wt")
            bigbed_infilename = temp_file.name
            outfile = Tee(outfile, temp_file)

        with outfile as outfile:
            ends = self.write(outfile)

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
                with NamedTemporaryFile(prefix=__package__, suffix=SUFFIX_TAB, mode="wt") as sizes_file:
                    for chrom, end in viewitems(ends):
                        print(chrom, str(end), sep="\t", file=sizes_file)

                    sizes_file.flush()
                    BEDTOBIGBED_PROG(bigbed_infilename, sizes_file.name, bigbed_outfilename)


    def write_trackline(self, outfile):
        # If self.trackline is None then TypeErrors occur when joining/printing.
        # This assertion can help single out trackline if it is the culprit
        assert(self.trackline)
        try:
            final_outfile = outfile._items[0]
        except AttributeError:
            final_outfile = outfile
        print(" ".join(self.trackline), file=final_outfile)

    def write(self, outfile):
        ends = {}
        mnemonics = self.mnemonics
        labels_sorted = self.labels_sorted
        label_dict = self.label_dict
        colors = self.colors

        self.write_trackline(outfile)

        for chrom, chromosome in sorted(viewitems(self)):
            for run in chromosome:
                segments = array(run)

                start = segments[0, OFFSET_START]
                end = segments[-1, OFFSET_END]
                ends[chrom] = end

                for label in labels_sorted:
                    label_key = label_dict[label]
                    color = colors.get(label, COLOR_DEFAULT)

                    # find all the rows for this label
                    segments_label_rows = segments[:, OFFSET_LABEL] == label_key

                    # extract just the starts and ends
                    segments_label = segments[segments_label_rows,
                                              OFFSET_START:OFFSET_END+1]

                    # pad end if necessary
                    segments_label_list = [segments_label]
                    if not len(segments_label) or segments_label[-1, OFFSET_END] != end:
                        # must be end-1 to end or UCSC gets cranky.
                        # unfortunately this results in all on at the
                        # right edge of each region
                        segments_label_list.append((end-1, end))

                    # pad beginning if necessary
                    if not len(segments_label) or segments_label[0, OFFSET_START] != start:
                        segments_label_list.insert(0, (start, start+1))

                    segments_label = vstack(segments_label_list)

                    # reverse offset by start
                    segments_label -= start

                    block_count = str(len(segments_label))

                    block_sizes = diff(segments_label).ravel()
                    block_sizes_str = make_csv(block_sizes)

                    block_starts = segments_label[:, 0]
                    block_starts_str = make_csv(block_starts)

                    # this just passes through the label itself if there
                    # are no mnemonics
                    mnemonic = mnemonics[str(label)]

                    row = [chrom, str(start), str(end), mnemonic, BED_SCORE,
                           BED_STRAND, str(start), str(end), color, block_count,
                           block_sizes_str, block_starts_str]

                    print(*row, sep="\t", file=outfile)

        return ends

def layer(infilename="-", outfilename="-", mnemonic_filename=None,
          trackline_updates={}, bigbed_outfilename=None, do_recolor=False):
    segmentation = Segmentation()
    segmentation.load_mnemonics(mnemonic_filename)
    segmentation.load(infilename)
    segmentation.update_trackline(trackline_updates)

    if do_recolor:
        segmentation.recolor()

    segmentation.save(outfilename, bigbed_outfilename)

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
    parser.add_option("--no-recolor", action="store_false",
                      dest="recolor", default=True,
                      help="don't recolor labels")
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
                 do_recolor=options.recolor,
                 *args)

if __name__ == "__main__":
    sys.exit(main())
