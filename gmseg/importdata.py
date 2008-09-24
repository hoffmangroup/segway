#!/usr/bin/env python
from __future__ import division, with_statement

"""
importdata: DESCRIPTION

each SRC is either
- *.list: a newline-delimited list of files
  - each file is a BED file which has a single observation
- *.wigVar.gz: a variableStep wiggle file
- *.txt.gz: tabular representation of wiggle track (described in
   unused .sql file)

DST is a directory, which will contain one HDF5 file for each chrom in the data
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

from collections import defaultdict
from functools import partial
from gzip import open as gzip_open
from os import extsep
import sys

from numpy import NAN
from path import path
from tabdelim import DictReader
from tables import Float64Atom, NoSuchNodeError, openFile

from .bed import read_native
from ._util import walk_supercontigs

ATOM = Float64Atom(dflt=NAN)

EXT_H5 = "h5"

KEYEQ_CHROM = "chrom="
LEN_KEYEQ_CHROM = len(KEYEQ_CHROM)

DEFAULT_WIGVAR_PARAMS = dict(span=1)

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

def pairs2dict(texts):
    res = {}

    for text in texts:
        text_part = text.partition("=")
        res[text_part[0]] = text_part[2]

    return res

def chromosome_factory(outdirpath, key):
    # only runs when there is not already a dictionary entry
    filename = outdirpath / extsep.join([key, EXT_H5])

    return openFile(filename, "r+")

def write_score(chromosome, start, end, score):
    # XXXopt: do profiling first:
    # XXXopt: some caching and lazy-writing of all this would be
    # good; in most cases, the chromosome will be the same
    # throughout

    for supercontig in walk_supercontigs(chromosome):
        attrs = supercontig._v_attrs
        supercontig_start = attrs.start
        supercontig_end = attrs.end

        if supercontig_start <= start and supercontig_end >= end:
            break
    else:
        raise DataForGapError("%r does not fit into a single supercontig"
                              % datum)

    try:
        continuous = supercontig.continuous
    except NoSuchNodeError:
        shape = (supercontig_end - supercontig_start, num_cols)
        continuous = chromosome.createCArray(supercontig, "continuous", ATOM,
                                             shape)

    row_start = start - supercontig_start
    row_end = end - supercontig_start
    continuous[row_start:row_end, col_index] = score

def read_bed(col_index, bedfile, num_cols, chromosomes):
    for datum in read_native(bedfile):
        chromosome = chromosomes[datum.chrom]

        start = datum.chromStart
        end = datum.chromEnd
        score = datum.score

        write_score(chromosome, start, end, score)

def read_filelist(filename_index, infile, num_cols, chromosomes):
    for line in infile:
        filename = line.rstrip()
        print >>sys.stderr, filename

        # recursion, whee!
        load_any(filename_index, filename, num_cols, chromosomes)

def read_wigvar(filename_index, infile, num_cols, chromosomes):
    chromosome = None
    span = None

    for line in infile:
        words = line.rstrip().split()
        num_words = len(words)

        # fixedStep not supported
        if word[0] == "variableStep":
            params = DEFAULT_WIGVAR_PARAMS.copy()
            params.update(pairs2dict(words[1:]))

            chromosome = chromosomes[params["chrom"]]
            span = int(params["span"])
        else:
            assert num_words == 2

            start = int(words[0]) - 1 # one-based
            end = start + span
            score = float(words[1])

            write_score(chromosome, start, end, score)

def read_mysql_tab(filename_index, infile, num_cols, chromosomes):
    for row in DictReader(infile, FIELDNAMES_MYSQL_TAB):
        chromosome = chromosomes[row["chrom"]]
        start = int(row["chromStart"])
        end = int(row["chromEnd"])
        score = float(row["dataValue"])

        write_score(chromosome, start, end, score)

def read_any(filename_index, filename, infile, num_cols, chromosomes):
    if filename.endswith(".list"):
        reader = read_filelist
    elif filename.endswith(".bed"):
        reader = read_bed
    elif filename.endswith(".wigVar"):
        reader = read_wigvar
    elif filename.endswith(".txt"):
        reader = read_mysql_tab
    else:
        raise ValueError, "file extension not recognized"

    return reader(filename_index, infile, num_cols, chromosomes)

def load_uncompressed(filename_index, filename, num_cols, chromosomes):
    with open(filename) as infile:
        read_any(filename_index, filename, infile, num_cols, chromosomes):

def load_gzip(filename_index, filename, num_cols, chromosomes):
    # remove .gz for further type sniffing
    filename_stem = filename.rpartition(extsep)[0]

    with gzip_open(filename) as infile:
        return read_any(filename_index, filename_stem, infile, num_cols,
                        chromosomes)

def load_any(filename_index, filename, num_cols, chromosomes):
    if filename.endswith(".gz"):
        return load_gzip(filename_index, filename, num_cols, chromosomes)
    else:
        return load_uncompressed(filename_index, filename, num_cols,
                                 chromosomes)
def importdata(filenames, outdirname):
    outdirpath = path(outdirname)
    num_cols = len(filenames)

    configured_chromosome_factory = partial(chromosome_factory, outdirpath)
    chromosomes = KeyPassingDefaultdict(configured_chromosome_factory)

    try:
        for filename_index, filename in enumerate(filenames):
            load_any(filename_index, filename, num_cols, chromosomes)
    finally:
        for chromosome in chromosomes.itervalues():
            chromosome.close()

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... SRC... DST"
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)

    options, args = parser.parse_args(args)

    if not len(args) >= 2:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return importdata(args[:-1], args[-1])

if __name__ == "__main__":
    sys.exit(main())
