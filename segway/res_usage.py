#!/usr/bin/env python
from __future__ import division

"""
res_usage: measure resource_usage
"""

__version__ = "$Revision$"

# Copyright 2009 Michael M. Hoffman <mmh1@washington.edu>

from os import getpid
import sys

from numpy import arange, square
from numpy.random import standard_normal

from .run import Runner
from ._utils import fill_array

MAX_NUM_TRACKS = 20
NUM_OBSERVATIONS = 1000000

CHROM_FAKE = "fake"

DIR_FMT = "res_usage_%d"
TRACKNAME_FMT = "obs%d"
HUGE_MEM_REQ = "10G"

class MemUsageRunner(Runner):
    """
    finds memory usage instead of using real data
    """
    def write_observations(self, float_filelist, int_filelist):
        num_tracks = self.num_tracks

        # from Runner.set_tracknames()
        tracknames = [TRACKNAME_FMT % track_index
                      for track_index in xrange(num_tracks)]
        self.tracknames = tracknames
        self.tracknames_all = tracknames
        self.track_indexes = arange(len(tracknames))

        # make files
        float_filepath, int_filepath = self.print_obs_filepaths(float_filelist,
                                                                int_filelist,
                                                                CHROM_FAKE, 0)

        cells = standard_normal((NUM_OBSERVATIONS, num_tracks))

        # from Runner.accum_metadata()
        row_shape = (NUM_OBSERVATIONS,)
        self.mins = cells.min(0)
        self.maxs = cells.max(0)
        self.sums = cells.sum(0)
        self.sums_squares = square(cells).sum(0)
        self.num_datapoints = fill_array(cells.shape[0], row_shape)

        self.save_observations_chunk(float_filepath, int_filepath, cells, None)

        self.num_int_cols = num_tracks
        self.num_chunks = 1 # a "chunk" is what GMTK calls a segment
        self.num_bases = NUM_OBSERVATIONS
        self.chunk_coords = [(CHROM_FAKE, 0, NUM_OBSERVATIONS)]

    @staticmethod
    def make_mem_req(chunk_len, num_tracks):
        return HUGE_MEM_REQ

    def make_job_name_train(self, start_index, round_index, chunk_index):
        chunk_len = self.chunk_lens[chunk_index]
        return "ru%s.%d.%d" % (getpid(), self.num_tracks, chunk_len)

def res_usage():
    for num_tracks in xrange(1, MAX_NUM_TRACKS):
        runner = MemUsageRunner()
        runner.dirname = DIR_FMT % num_tracks
        runner.num_tracks = num_tracks
        runner.identify = False

        return runner()

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]..."
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)

    options, args = parser.parse_args(args)

    if not len(args) == 0:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return res_usage(*args)

if __name__ == "__main__":
    sys.exit(main())
