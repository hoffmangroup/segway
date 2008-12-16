#!/usr/bin/env python
from __future__ import division, with_statement

"""
save_metadata: DESCRIPTION
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

import sys

from numpy import amin, amax, array, isnan, NINF, PINF, where
from tables import openFile

from .load_seq import MIN_GAP_LEN
from ._util import (fill_array, get_tracknames, init_num_obs, new_extrema,
                    walk_continuous_supercontigs)

def write_metadata(chromosome):
    print >>sys.stderr, "writing metadata for %s" % chromosome.title

    num_obs = len(get_tracknames(chromosome))
    extrema_shape = (num_obs,)
    mins = fill_array(PINF, extrema_shape)
    maxs = fill_array(NINF, extrema_shape)

    for supercontig, continuous in walk_continuous_supercontigs(chromosome):
        # only runs when assertions checked
        if __debug__:
            init_num_obs(num_obs, continuous) # for the assertion

        ## read data
        observations = continuous.read()
        mask_missing = isnan(observations)

        observations[mask_missing] = PINF
        mins = new_extrema(amin, observations, mins)

        observations[mask_missing] = NINF
        maxs = new_extrema(amax, observations, maxs)

        ## find chunks that have less than MIN_GAP_LEN missing data
        ## gaps in a row
        rows_num_missing = mask_missing.sum(1)
        mask_rows_any_nonmissing = rows_num_missing < num_obs
        indices_nonmissing = where(mask_rows_any_nonmissing)[0]

        starts = []
        ends = []

        # so that index - last_index is always >= MIN_GAP_LEN
        last_index = -MIN_GAP_LEN
        for index in indices_nonmissing:
            if index - last_index >= MIN_GAP_LEN:
                if starts:
                    # add 1 because we want slice(start, end) to
                    # include the last_index
                    ends.append(last_index + 1)

                starts.append(index)
            last_index = index

        if last_index >= 0:
            ends.append(last_index + 1)

        assert len(starts) == len(ends)

        if not starts:
            # remove empty supercontigs continuous
            continuous._f_remove()

        supercontig_attrs = supercontig._v_attrs
        supercontig_attrs.chunk_starts = array(starts)
        supercontig_attrs.chunk_ends = array(ends)

    chromosome_attrs = chromosome.root._v_attrs
    chromosome_attrs.mins = mins
    chromosome_attrs.maxs = maxs
    chromosome_attrs.dirty = False

def save_metadata(*filenames):
    for filename in filenames:
        with openFile(filename, "r+") as chromosome:
            write_metadata(chromosome)

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... FILE..."
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)

    options, args = parser.parse_args(args)

    if not len(args) >= 1:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return save_metadata(*args)

if __name__ == "__main__":
    sys.exit(main())
