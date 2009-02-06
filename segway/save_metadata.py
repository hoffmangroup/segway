#!/usr/bin/env python
from __future__ import division, with_statement

"""
save_metadata: DESCRIPTION
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

import sys

from numpy import amin, amax, append, array, diff, isfinite, NINF, PINF, square
from tables import openFile

from .load_seq import MIN_GAP_LEN
from ._util import (fill_array, get_tracknames, init_num_obs, new_extrema,
                    walk_continuous_supercontigs)

def update_extrema(func, extrema, data, col_index):
    extrema[col_index] = new_extrema(func, data, extrema[col_index])

def write_metadata(chromosome):
    print >>sys.stderr, "writing metadata for %s" % chromosome.title

    tracknames = get_tracknames(chromosome)
    num_obs = len(tracknames)
    row_shape = (num_obs,)
    mins = fill_array(PINF, row_shape)
    maxs = fill_array(NINF, row_shape)
    sums = fill_array(0.0, row_shape)
    sums_squares = fill_array(0.0, row_shape)
    num_datapoints = fill_array(0, row_shape)

    for supercontig, continuous in walk_continuous_supercontigs(chromosome):
        print >>sys.stderr, " scanning %s" % supercontig._v_name

        # only runs when assertions checked
        if __debug__:
            init_num_obs(num_obs, continuous) # for the assertion

        num_rows = continuous.shape[0]
        mask_rows_any_present = fill_array(False, num_rows)

        # doing this column by column greatly reduces the memory
        # footprint when you have large numbers of tracks. It also
        # simplifies the logic for the summary stats, since you don't
        # have to change the mask value for every operation, like in
        # revisions <= r243
        for col_index, trackname in enumerate(tracknames):
            print >>sys.stderr, "  %s" % trackname

            ## read data
            col = continuous[:, col_index]

            mask_present = isfinite(col)
            mask_rows_any_present[mask_present] = True
            col_finite = col[mask_present]
            # XXXopt: should be able to overwrite col, not needed anymore

            num_datapoints_col = len(col_finite)
            if num_datapoints_col:
                update_extrema(amin, mins, col_finite, col_index)
                update_extrema(amax, maxs, col_finite, col_index)

                sums[col_index] += col_finite.sum(0)
                sums_squares[col_index] += square(col_finite).sum(0)
                num_datapoints[col_index] += num_datapoints_col

        ## find chunks that have less than MIN_GAP_LEN missing data
        ## gaps in a row

        # get all of the indices where there is any data
        indices_present = mask_rows_any_present.nonzero()[0]

        if not len(indices_present):
            # remove continuous of empty supercontigs
            continuous._f_remove()
            continue

        # make a mask of whether the difference from one index to the
        # next is >= MIN_GAP_LEN
        diffs_signif = diff(indices_present) >= MIN_GAP_LEN

        # convert the mask back to indices of the original indices
        indices_signif = diffs_signif.nonzero()[0]

        if len(indices_signif):
            starts = indices_present[indices_signif]

            # finish with the index immediately before each start, and the
            # last index
            ends = indices_present[append(indices_signif[1:]-1, -1)]

            # add 1 because we want slice(start, end) to include the
            # last_index
            ends += 1
        else:
            starts = array(0)
            ends = array(num_rows)

        supercontig_attrs = supercontig._v_attrs
        supercontig_attrs.chunk_starts = starts
        supercontig_attrs.chunk_ends = ends

    chromosome_attrs = chromosome.root._v_attrs
    chromosome_attrs.mins = mins
    chromosome_attrs.maxs = maxs
    chromosome_attrs.sums = sums
    chromosome_attrs.sums_squares = sums_squares
    chromosome_attrs.num_datapoints = num_datapoints
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
