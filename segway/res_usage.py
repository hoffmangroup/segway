#!/usr/bin/env python
from __future__ import division

"""
res_usage: measure resource_usage
"""

__version__ = "$Revision$"

# Copyright 2009 Michael M. Hoffman <mmh1@washington.edu>

from os import getpid
import sys

from numpy import arange, float32, square
from numpy.random import standard_normal

from .run import Runner
from ._util import fill_array

MAX_NUM_TRACKS = 20
MIN_EXPONENT = 4
MAX_EXPONENT = 6

CHROM_FMT = "fake%d"

DIR_FMT = "res_usage_%d"
TRACKNAME_FMT = "obs%d"
HUGE_MEM_REQ = "5G" # 4G will fit 1e6*20 cells

class MemUsageRunner(Runner):
    """
    finds memory usage instead of using real data
    """
    def __init__(self, *args, **kwargs):
        Runner.__init__(self, *args, **kwargs)
        self.identify = False

    def write_observations(self, float_filelist, int_filelist):
        num_tracks = self.num_tracks

        # from Runner.set_tracknames()
        tracknames = [TRACKNAME_FMT % track_index
                      for track_index in xrange(num_tracks)]
        self.tracknames = tracknames
        self.tracknames_all = tracknames
        self.track_indexes = arange(len(tracknames))
        self.num_int_cols = num_tracks

        chunk_coords = [] # a "chunk" is what GMTK calls a segment
        num_bases = 0

        num_observations_list = []
        exponent_range = xrange(MAX_EXPONENT-1, MIN_EXPONENT-1, -1)
        for exponent in exponent_range:
            num_observations_array = arange(10, 1, -1, int) * 10**exponent
            num_observations_list += num_observations_array.tolist()

        for chunk_index, num_observations in enumerate(num_observations_list):
            chrom = CHROM_FMT % num_observations

            print >>sys.stderr, (num_tracks, num_observations)

            # make files
            float_filepath, int_filepath = \
                self.print_obs_filepaths(float_filelist, int_filelist, chrom,
                                         chunk_index)

            cells = float32(standard_normal((num_observations, num_tracks)))

            # XXX: refactor into accum_metadata_calc():
            # """recalculates metadata from real data"""
            row_shape = (num_tracks,)
            mins = cells.min(0)
            maxs = cells.max(0)
            sums = cells.sum(0)
            sums_squares = square(cells).sum(0)
            num_datapoints = fill_array(cells.shape[0], row_shape)
            self.accum_metadata(mins, maxs, sums, sums_squares, num_datapoints)

            self.save_observations_chunk(float_filepath, int_filepath, cells,
                                         None)

            num_bases += num_observations
            chunk_coords.append((chrom, 0, num_observations))

        self.num_chunks = chunk_index
        self.num_bases = num_bases
        self.chunk_coords = chunk_coords

    @staticmethod
    def make_mem_req(chunk_len, num_tracks):
        return HUGE_MEM_REQ

    def make_job_name_train(self, start_index, round_index, chunk_index):
        chunk_len = self.chunk_lens[chunk_index]

        return "ru%s.%d.%d" % (getpid(), self.num_tracks, chunk_len)

    def run_train_round(self, start_index, round_index, **kwargs):
        # just run all the parallel jobs, no bundle
        self.queue_train_parallel(self.last_params_filename, start_index,
                                  round_index, **kwargs)

        # cause loop to abort
        return False

    def proc_train_results(self, start_params, dst_filenames):
        # don't do any processing
        return

def res_usage():
    for num_tracks in xrange(MAX_NUM_TRACKS, 0, -1):
        runner = MemUsageRunner()
        runner.dirname = DIR_FMT % num_tracks
        runner.num_tracks = num_tracks
        runner.delete_existing = True

        runner()

    print "PID: %s" % getpid()
    # XXX: need something like this after everything is done:
    # with Session(), etc.
    # qsub -sync y -b y -hold_jid "ru18914.*" -cwd -o ru18914.txt $(which qacct) -j "\"ru18914.*\""

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
