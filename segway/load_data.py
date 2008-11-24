#!/usr/bin/env python
from __future__ import division, with_statement

"""
load_data: DESCRIPTION

each SRC is either
- *.list: a newline-delimited list of files
  - each file is a BED file which has a single observation
- *.wigVar.gz, *.wigFix.gz, *.wig.gz, *.pp.gz: a wiggle file
- *.txt.gz: tabular representation of wiggle track (described in
   unused .sql file)

DST is a directory, which will contain one HDF5 file for each chrom in the data
"""

# XXXopt: the biggest bottlenecks are gzip reading and duplicating the
# values for the numpy array. There is a Python patch to take care of
# the first, I don't know if there is anything useful to be done for
# the second

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

from collections import defaultdict
from functools import partial
from os import extsep
import sys
from warnings import warn

from numpy import float32, NAN, NINF, PINF
from path import path
from tabdelim import DictReader
from tables import Float32Atom, NoSuchNodeError, openFile

#from .bed import read_native
from ._util import get_tracknames, gzip_open, walk_supercontigs

ATOM = Float32Atom(dflt=NAN)

EXT_H5 = "h5"

KEYEQ_CHROM = "chrom="
LEN_KEYEQ_CHROM = len(KEYEQ_CHROM)

DEFAULT_WIG_PARAMS = dict(span=1)
WIG_FMT_SET = set(["variableStep", "fixedStep"])

# XXX: hacky: should really load into a real MySQL database instead
FIELDNAMES_MYSQL_TAB = ["bin", "chrom", "chromStart", "chromEnd", "dataValue"]

class DataForGapError(ValueError): pass

class KeyPassingDefaultdict(defaultdict):
    """
    like defaultdict, but default_factory is passed key
    """
    def __missing__(self, key):
        default_factory = self.default_factory
        if default_factory is None:
            raise KeyError, key

        self[key] = res = default_factory(key)

        return res

class ScoreWriter(object):
    """
    caches current supercontig before writing
    """
    def __init__(self, trackname):
        self.trackname = trackname
        self._clear()
        self._write = self._write_flexible

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.flush()

    @staticmethod
    def _get_supercontig_coords(supercontig):
        attrs = supercontig._v_attrs
        return attrs.start, attrs.end

    def _seek_pos(self, pos):
        chromosome = self.chromosome

        for supercontig in walk_supercontigs(chromosome):
            supercontig_start, supercontig_end = \
                self._get_supercontig_coords(supercontig)
            if supercontig_start <= pos < supercontig_end:
                return supercontig

        return None

    def _seek(self, start, end=None):
        supercontig = self._seek_pos(start)
        if supercontig is None:
            if end is None:
                end = start + 1

            supercontig = self._seek_pos(end)

            if supercontig is None:
                # hopefully there won't be cases where *both* ends
                # won't fit, but the middle does

                # an exception error leaves all of the instance
                # attributes untouched
                raise DataForGapError("neither %d nor %d fit into a"
                                      " supercontig" % (start, end))

        supercontig_start, supercontig_end = \
            self._get_supercontig_coords(supercontig)

        self.flush()

        try:
            continuous = supercontig.continuous
        except NoSuchNodeError:
            createCArray = self.chromosome.createCArray
            shape = (supercontig_end - supercontig_start, self.num_cols)
            continuous = createCArray(supercontig, "continuous", ATOM, shape)

        self.continuous = continuous
        self.continuous_array = continuous[..., self.col_index]
        self.start = supercontig_start
        self.end = supercontig_end

        return supercontig_start

    def _clear(self):
        """
        clears attributes that are chromosome-specific
        """
        self.chromosome = None
        self.col_index = None
        self.num_cols = None
        self.continuous = None
        self.continuous_array = None
        self.start = PINF
        self.end = NINF

    def _write_flexible(self, score, start, end=None):
        # if you try to write to (400, 405) and the supercontig only
        # goes up to 401, then data will be written to 400 and 401
        # without error. if you try to write to (402, 405), then you
        # will get an exception
        supercontig_start = self.start

        if end is None:
            end = start + self.span

        if self.end <= start or start < supercontig_start:
            supercontig_start = self._seek(start, end)

            # if it is still off the left edge
            start = max(start, supercontig_start)

        row_start = start - supercontig_start
        row_end = end - supercontig_start

        self.continuous_array[row_start:row_end] = score

    def _write_span1(self, score, start):
        # this special-case optimization results in a speed increase of 27%
        # but switching to a different function only gets you 3% of that
        supercontig_start = self.start
        row_start = start - supercontig_start

        # < 0 means start < supercontig_start
        # supposedly can't happen
        if self.end <= start or row_start < 0:
            row_start = max(start - self._seek(start), 0)

        # most of the optimization is in not using a slice here:
        self.continuous_array[row_start] = score

    def set_chromosome(self, chromosome):
        if chromosome != self.chromosome:
            self.flush()
            self._clear()
            self.chromosome = chromosome

            tracknames = get_tracknames(chromosome)
            self.num_cols = len(tracknames)
            self.col_index = tracknames.index(self.trackname)

    def set_span(self, span):
        self.span = span
        if span == 1:
            self._write = self._write_span1
        else:
            self._write = self._write_flexible

    def write(self, *args, **kwargs):
        try:
            return self._write(*args, **kwargs)
        except DataForGapError:
            warn("data supplied outside supercontig: %r %r" % (args, kwargs))

    def flush(self):
        """
        do actual writing; stop being lazy
        """
        continuous = self.continuous
        if continuous:
            continuous[..., self.col_index] = self.continuous_array

def pairs2dict(texts):
    res = {}

    for text in texts:
        text_part = text.partition("=")
        res[text_part[0]] = text_part[2]

    return res

def chromosome_factory(outdirpath, key):
    # only runs when there is not already a dictionary entry
    filename = outdirpath / extsep.join([key, EXT_H5])

    res = openFile(filename, "r+")
    res.root._v_attrs.dirty = True

    return res

def write_score(chromosome, start, end, score, col_index, num_cols):
    raise NotImplementedError, "should eliminate all callers"

##    # XXX: remove calls to this; call ScoreWriter directly in parent
##
##    with ScoreWriter(col_index, num_cols) as writer XXX:
##        writer.set_chromosome(chromosome)
##        writer.write(start, end, score)

def read_bed(chromosomes, trackname, filename, infile):
    raise NotImplementedError, "need to update code for trackname regime; " \
        "replace write_score with direct call to ScoreWriter"

##    for datum in read_native(infile):
##        chromosome = chromosomes[datum.chrom]
##
##        start = datum.chromStart
##        end = datum.chromEnd
##        score = datum.score
##
##        write_score(chromosome, start, end, score, col_index, num_cols)

def read_filelist(chromosomes, trackname, filename, infile):
    for line in infile:
        filename = line.rstrip()

        # recursion
        load_any(chromosomes, trackname, filename, infile)

def read_wig(chromosomes, trackname, filename, infile):
    chromosome = None
    start = None
    step = None
    span = None
    fmt = None

    with ScoreWriter(trackname) as writer:
        for index, line in enumerate(infile):
            if not index % 100000:
                print >>sys.stderr, ("[%d]" % index),

            words = line.rstrip().split()
            num_words = len(words)

            if words[0] in WIG_FMT_SET:
                fmt = words[0]

                params = DEFAULT_WIG_PARAMS.copy()
                params.update(pairs2dict(words[1:]))

                chrom = params["chrom"]
                chromosome = chromosomes[chrom]
                writer.set_chromosome(chromosome)

                span = int(params["span"])
                writer.set_span(span)

                if fmt == "fixedStep":
                    start = int(params["start"]) - 1 # one-based
                    step = int(params["step"])

                    print >>sys.stderr, "\n%s (%d)" % (chrom, start)
                else:
                    assert "start" not in params
                    assert "step" not in params

                    start = None
                    step = None

                    print >>sys.stderr, "\n%s" % chrom
            elif fmt == "variableStep":
                assert num_words == 2

                start = int(words[0]) - 1 # one-based
                score = float32(words[1])
                writer.write(score, start)
            elif fmt == "fixedStep":
                # XXXopt: probably could get a speedup by reading in
                # multiple lines before writing. read up until the end
                # of the current supercontig
                assert num_words == 1

                score = float32(words[0])
                writer.write(score, start)

                start += step
            elif words[0] == "track":
                # just ignore these lines
                pass
            else:
                raise ValueError, "only fixedStep and variableStep formats " \
                    " are supported"

def read_mysql_tab(chromosomes, trackname, filename, infile):
    with ScoreWriter(trackname) as writer:
        for row in DictReader(infile, FIELDNAMES_MYSQL_TAB):
            chromosome = chromosomes[row["chrom"]]
            writer.set_chromosome(chromosome)

            start = int(row["chromStart"])
            end = int(row["chromEnd"])
            score = float32(row["dataValue"])

            writer.write(score, start, end)

READERS = dict(list=read_filelist,
               bed=read_bed,
               pp=read_wig,
               wigFix=read_wig,
               wigVar=read_wig,
               wig=read_wig,
               txt=read_mysql_tab)

def read_any(chromosomes, trackname, filename, infile):
    ext = filename.rpartition(".")[2]

    try:
        reader = READERS[ext]
    except KeyError:
        raise ValueError, "file extension not recognized"

    return reader(chromosomes, trackname, filename, infile)

def load_uncompressed(chromosomes, trackname, filename):
    with open(filename) as infile:
        return read_any(chromosomes, trackname, filename, infile)

def load_gzip(chromosomes, trackname, filename):
    # remove .gz for further type sniffing
    filename_stem = filename.rpartition(extsep)[0]

    with gzip_open(filename) as infile:
        return read_any(chromosomes, trackname, filename_stem, infile)

def load_any(chromosomes, trackname, filename):
    print >>sys.stderr, filename

    if filename.endswith(".gz"):
        return load_gzip(chromosomes, trackname, filename)
    else:
        return load_uncompressed(chromosomes, trackname, filename)

def load_data(outdirname, trackname, *filenames):
    outdirpath = path(outdirname)

    configured_chromosome_factory = partial(chromosome_factory, outdirpath)
    chromosomes = KeyPassingDefaultdict(configured_chromosome_factory)

    try:
        for col_index, filename in enumerate(filenames):
            load_any(chromosomes, trackname, filename)
    finally:
        for chromosome in chromosomes.itervalues():
            chromosome.close()

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... DST TRACKNAME SRC..."
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)

    options, args = parser.parse_args(args)
    # XXX: add options to refresh metadata only (do not destroy on openFile!)

    if not len(args) >= 3:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return load_data(*args)

if __name__ == "__main__":
    sys.exit(main())
