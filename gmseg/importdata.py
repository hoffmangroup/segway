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
from errno import EEXIST
from os import extsep
import sys

from numpy import append, insert, zeros
from path import path
from tables import Filters, Float64Atom, NoSuchNodeError, openFile

from .bed import read_native

SUPERCONTIG_ONLY = "only"
NAME_CONTINUOUS = "continuous"
ATOM = Float64Atom()

EXT_H5 = "h5"

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

def chromosome_metafactory(outdirpath, num_cols):
    def chromosome_factory(key):
        filename = outdirpath / extsep.join([key, EXT_H5])
        filters = Filters(complevel=1)
        res = openFile(filename, "w", key, filters=filters)

        supercontig = res.createGroup("/", SUPERCONTIG_ONLY) # supercontig
        supercontig._v_attrs.offset = None

        shape = (0, num_cols)
        res.createEArray(supercontig, NAME_CONTINUOUS, ATOM, shape)

        return res

    return chromosome_factory

def import_bedfile(bedfile, chromosomes, col_index, num_cols):
    for datum in read_native(bedfile):
        # XXXopt: use EArrays

        # XXXopt: do profiling first:
        # XXXopt: some caching and lazy-writing of all this would be
        # good; in most cases, the chromosome will be the same
        # throughout

        chromosome = chromosomes[datum.chrom]
        supercontig = chromosome.getNode("/", SUPERCONTIG_ONLY)
        offset_start = supercontig._v_attrs.offset

        continuous = chromosome.getNode(supercontig, NAME_CONTINUOUS)

        start = datum.chromStart
        end = datum.chromEnd
        score = datum.score

        if offset_start is None:
            offset_start = start

            data = zeros((end - start, num_cols))
            data[:, col_index] = score
            continuous.append(data)
        else:
            offset_end = offset_start + continuous.shape[0]

            if start < offset_start:
                # extend the data with zeros, but do not write
                extended_data = zeros((offset_start - start, num_cols))
                data = insert(continuous.read(), 0, extended_data, 0)

                chromosome.removeNode(supercontig, NAME_CONTINUOUS)
                continuous = chromosome.createEArray(supercontig,
                                                     NAME_CONTINUOUS, ATOM,
                                                     (0, num_cols))
                continuous.append(data)

                offset_start = start
                assert offset_end == offset_start + data.shape[0]

            if end > offset_end:
                # extend the data with zeros, but do not write
                extended_data = zeros((end - offset_end, num_cols))
                data = append(data, extended_data, 0)

            # write
            continuous[start-offset_start:end-offset_start, col_index] = score

        supercontig._v_attrs.offset = offset_start

def importdata(filelistnames, outdirname):
    outdirpath = path(outdirname)
    try:
        outdirpath.makedirs()
    except OSError, err:
        if err.errno != EEXIST:
            raise

    num_cols = len(filelistnames)

    chromosome_factory = chromosome_metafactory(outdirpath, num_cols)
    chromosomes = KeyPassingDefaultdict(chromosome_factory)

    try:
        for filelist_index, filelistname in enumerate(filelistnames):
            with open(filelistname) as filelist:
                for line in filelist:
                    bedfilename = line.rstrip()

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
