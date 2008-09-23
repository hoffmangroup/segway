#!/usr/bin/env python
from __future__ import division, with_statement

"""
importdata: DESCRIPTION

each SRCLIST is a newline-delimited list of files
each file is a BED file which has a single observation

DST is a directory, which will contain one HDF5 file for each chrom in the data
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

from collections import defaultdict
from functools import partial
from os import extsep
import sys
from warnings import warn

from numpy import NAN
from path import path
from tables import Float64Atom, NoSuchNodeError, openFile

from .bed import read_native

ATOM = Float64Atom(dflt=NAN)

EXT_H5 = "h5"

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

def chromosome_factory(outdirpath, key):
    # only runs when there is not already a dictionary entry
    filename = outdirpath / extsep.join([key, EXT_H5])

    return openFile(filename, "r+")

def import_bedfile(bedfile, chromosomes, col_index, num_cols):
    for datum in read_native(bedfile):
        # XXXopt: use EArrays

        # XXXopt: do profiling first:
        # XXXopt: some caching and lazy-writing of all this would be
        # good; in most cases, the chromosome will be the same
        # throughout

        chromosome = chromosomes[datum.chrom]
        root = chromosome.root

        start = datum.chromStart
        end = datum.chromEnd
        score = datum.score

        for supercontig in chromosome.walkGroups():
            if supercontig == root: # not really a supercontig
                continue

            attrs = supercontig._v_attrs
            supercontig_start = attrs.start
            supercontig_end = attrs.end

            if supercontig_start <= start and supercontig_end >= end:
                break
        else:
            warn("%r does not fit into a single supercontig" % datum)
            continue

        try:
            continuous = supercontig.continuous
        except NoSuchNodeError:
            shape = (supercontig_end - supercontig_start, num_cols)
            continuous = chromosome.createCArray(supercontig, "continuous",
                                                 ATOM, shape)

        row_start = start - supercontig_start
        row_end = end - supercontig_start
        continuous[row_start:row_end, col_index] = score

def importdata(filelistnames, outdirname):
    outdirpath = path(outdirname)
    num_cols = len(filelistnames)

    configured_chromosome_factory = partial(chromosome_factory, outdirpath)
    chromosomes = KeyPassingDefaultdict(configured_chromosome_factory)

    try:
        for filelist_index, filelistname in enumerate(filelistnames):
            with open(filelistname) as filelist:
                for line in filelist:
                    bedfilename = line.rstrip()
                    print >>sys.stderr, bedfilename

                    with open(bedfilename) as bedfile:
                        import_bedfile(bedfile, chromosomes,
                                       filelist_index, num_cols)
    finally:
        for chromosome in chromosomes.itervalues():
            chromosome.close()

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... SRCLIST... DST"
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
