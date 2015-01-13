#!/usr/bin/env python
from __future__ import division

"""observations.py: prepare and save GMTK observations
"""

__version__ = "$Revision$"

## Copyright 2012, 2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from cStringIO import StringIO
from collections import deque
from contextlib import closing
from functools import partial
from itertools import izip, repeat
from os import extsep
import sys
from tempfile import gettempdir

from numpy import (add, append, arange, arcsinh, array, column_stack, empty,
                   invert, isnan, maximum, zeros)
from path import path
from tabdelim import ListWriter

from ._util import (ceildiv, copy_attrs, DISTRIBUTION_ASINH_NORMAL,
                    DTYPE_OBS_INT, EXT_FLOAT, EXT_INT, extjoin,
                    get_chrom_coords, make_prefix_fmt,
                    SUPERVISION_LABEL_OFFSET, SUPERVISION_SEMISUPERVISED,
                    SUPERVISION_UNSUPERVISED, USE_MFSDG, Window)

# number of frames in a segment must be at least number of frames in model
MIN_FRAMES = 2

MAX_WINDOWS = 9999

FLOAT_TAB_FIELDNAMES = ["filename", "window_index", "chrom", "start", "end"]

ORD_A = ord("A")
ORD_C = ord("C")
ORD_G = ord("G")
ORD_T = ord("T")
ORD_a = ord("a")
ORD_c = ord("c")
ORD_g = ord("g")
ORD_t = ord("t")


class NoData(object):
    """
    sentinel for not adding an extra field to coords, so that one can
    still use None
    """


def convert_windows(attrs, name):
    supercontig_start = attrs.start
    edges_array = getattr(attrs, name) + supercontig_start

    return edges_array.tolist()


def update_starts(starts, ends, new_starts, new_ends):
    # reversed because extend left extends deque in reversed order
    starts.extendleft(reversed(new_starts))
    ends.extendleft(reversed(new_ends))


def find_overlaps_include(start, end, coords, data=repeat(NoData)):
    """
    find items in coords that overlap (start, end)

    NOTE: multiple overlapping regions in coords will result in data
    being considered more than once
    """
    # diagram of how this works:
    #        --------------------- (include)
    #
    # various (start, end) cases:
    # A   --
    # B                              --
    # C   --------------------------
    # D   ------
    # E                       -------------
    # F             ---------
    res = []

    for (include_start, include_end), datum in izip(coords, data):
        if start > include_end or end <= include_start:
            # cases A, B
            continue
        elif start <= include_start:
            if end < include_end:
                # case D
                include_end = end

            # case C otherwise
        elif start > include_start:
            include_start = start

            if end < include_end:
                # case F
                include_end = end

            # case E otherwise
        else:
            assert False  # can't happen

        item = [include_start, include_end]

        if datum is not NoData:
            item.append(datum)

        res.append(item)

    return res


def find_overlaps_exclude(start, end, exclude_coords):
    """
    takes a start and end and removes everything in exclude_coords
    """
    # diagram of how this works:
    #        --------------------- (include)
    #
    # various exclude cases:
    # A   --
    # B                              --
    # C   --------------------------
    # D   ------
    # E                       -------------
    # F             ---------

    # does nothing if exclude_coords is empty

    if exclude_coords is None or len(exclude_coords) == 0:
        return [[start, end]]

    include_coords = [[start, end]]

    for exclude_start, exclude_end in exclude_coords:
        new_include_coords = []

        for include_coord in include_coords:
            start, end = include_coord

            if exclude_start > end or exclude_end <= start:
                # cases A, B
                new_include_coords.append([start, end])
            elif exclude_start <= start:
                if exclude_end >= end:
                    # case C
                    pass
                else:
                    # case D
                    new_include_coords.append([exclude_end, end])
            elif exclude_start > start:
                if exclude_end >= end:
                    # case E
                    new_include_coords.append([start, exclude_start])
                else:
                    # case F
                    new_include_coords.append([start, exclude_start])
                    new_include_coords.append([exclude_end, end])

            else:
                assert False  # can't happen

        include_coords = new_include_coords

    return include_coords


def calc_downsampled_shape(inarray, resolution):
    full_num_rows = inarray.shape[0]
    downsampled_num_rows = ceildiv(full_num_rows, resolution)
    return [downsampled_num_rows] + list(inarray.shape[1:])


def downsample_add(inarray, resolution):
    """
    Downsample a matrix by rows to a desired resolution, by adding up
    data points

    [1 2 4 8 16 32] downsampled to resolution 2 is:
    [1+2 4+8 16+32]

    originally: downsample presence data into num_datapoints
    now used for all downsampling

    inarray: array to be downsampled
    resolution: desired resolution

    """
    if resolution == 1:
        return inarray

    full_num_rows = inarray.shape[0]
    res = zeros(calc_downsampled_shape(inarray, resolution), inarray.dtype)

    # if there's no remainder, then only use loop 0
    remainder = full_num_rows % resolution
    if remainder == 0:
        remainder = resolution

    # loop 0: every index up to remainder
    for index in xrange(remainder):
        res += inarray[index::resolution]

    # loop 1: remainder
    for index in xrange(remainder, resolution):
        # don't include the last element of res
        res[:-1] += inarray[index::resolution]

    return res


def make_dinucleotide_int_data(seq):
    """
    makes an array with two columns, one with 0..15=AA..TT and the other
    as a presence variable. Set column one to 0 when not present
    """
    nucleotide_int_data = (((seq == ORD_A) + (seq == ORD_a))
                           + ((seq == ORD_C) + (seq == ORD_c)) * 2
                           + ((seq == ORD_G) + (seq == ORD_g)) * 3
                           + ((seq == ORD_T) + (seq == ORD_t)) * 4) - 1
    nucleotide_missing = nucleotide_int_data == -1

    # rewrite all Ns as A now that you have the missingness mask
    nucleotide_int_data[nucleotide_int_data == -1] = 0
    col_shape = (len(nucleotide_int_data) - 1,)

    # first column: dinucleotide: AA..TT=0..15
    # combine, and add extra AA stub at end, which will be set missing
    # 0 AA AC AG AT
    # 4 CA CC CG CT
    # 8 GA GC GG GT
    # 12 TA TC TG TT
    dinucleotide_int_data = empty(col_shape, DTYPE_OBS_INT)
    add(nucleotide_int_data[:-1] * 4, nucleotide_int_data[1:],
        dinucleotide_int_data)

    # second column: presence_dinucleotide: some missing=0; all present = 1
    # there are so few N boundaries that it is okay to
    # disregard the whole dinucleotide when half is N
    dinucleotide_missing = (nucleotide_missing[:-1] + nucleotide_missing[1:])

    dinucleotide_presence = empty(dinucleotide_missing.shape, DTYPE_OBS_INT)
    invert(dinucleotide_missing, dinucleotide_presence)

    # XXXopt: set these up properly in the first place instead of
    # column_stacking at the end
    return column_stack([dinucleotide_int_data, dinucleotide_presence])


def _save_window(float_filename, int_filename, float_data, resolution,
                 distribution, seq_data=None, supervision_data=None):
    # called by task.py as well as observation.py

    # input function in GMTK_ObservationMatrix.cc:
    # ObservationMatrix::readBinSentence

    # Input per frame is a series of float32s, followed by a series of
    # int32s. It is better to optimize both sides here by sticking all
    # the floats in one file, and the ints in another one.
    int_blocks = []
    if float_data is not None:
        if distribution == DISTRIBUTION_ASINH_NORMAL:
            float_data = arcsinh(float_data)

        if (not USE_MFSDG) or resolution > 1:
            mask_missing = isnan(float_data)

            # output -> presence_data -> int_blocks
            # done in two steps so I can specify output type
            presence_data = empty(mask_missing.shape, DTYPE_OBS_INT)
            invert(mask_missing, presence_data)

            num_datapoints = downsample_add(presence_data, resolution)

            # this is the presence observation
            int_blocks.append(num_datapoints)

            # so that there is no divide by zero
            num_datapoints_min_1 = maximum(num_datapoints, 1)

            # make float
            if not USE_MFSDG:
                float_data[mask_missing] = 0.0

            float_data = downsample_add(float_data, resolution)
            float_data /= num_datapoints_min_1

        float_data.tofile(float_filename)

    if seq_data is not None:
        assert resolution == 1  # not implemented yet
        int_blocks.append(make_dinucleotide_int_data(seq_data))

    if supervision_data is not None:
        assert resolution == 1  # not implemented yet
        int_blocks.append(supervision_data)

    if int_blocks:
        int_data = column_stack(int_blocks)
    else:
        int_data = array([], dtype=DTYPE_OBS_INT)

    int_data.tofile(int_filename)


def process_new_windows(new_windows, starts, ends):
    if not new_windows:  # nothing left
        return None, None
    elif len(new_windows) > 1:
        new_starts, new_ends = zip(*new_windows)
        update_starts(starts, ends, new_starts, new_ends)
        return None, None

    return new_windows[0]


class Observations(object):
    copy_attrs = ["include_coords", "exclude_coords", "max_frames",
                  "float_filelistpath", "int_filelistpath",
                  "float_tabfilepath", "obs_dirpath", "uuid", "resolution",
                  "distribution", "train", "identify", "supervision_type",
                  "supervision_coords", "supervision_labels",
                  "use_dinucleotide", "world_track_indexes", "clobber",
                  "num_worlds"]

    def __init__(self, runner):
        copy_attrs(runner, self, self.copy_attrs)

        self.float_filepaths = []
        self.int_filepaths = []

    def generate_coords_include(self):
        for chrom, coords_list in self.include_coords.iteritems():
            starts, ends = map(deque, zip(*coords_list))
            yield chrom, starts, ends

    def generate_coords_all(self, genome):
        for chromosome in genome:
            starts = deque()
            ends = deque()

            for supercontig, continuous in chromosome.itercontinuous():
                if continuous is not None:
                    attrs = supercontig.attrs

                    starts.extend(convert_windows(attrs, "chunk_starts"))
                    ends.extend(convert_windows(attrs, "chunk_ends"))

            if starts:
                yield chromosome.name, starts, ends

    def generate_coords(self, genome):
        """
        returns iterable of included coords, either explicitly
        specified, or all

          each item: tuple of (chrom, starts, ends)
          starts and ends are deques
        """
        if self.include_coords:
            return self.generate_coords_include()
        else:
            return self.generate_coords_all(genome)

    def skip_or_split_window(self, start, end):
        """
        skip short windows or skip long windows
        """
        max_frames = self.max_frames
        num_bases_window = end - start
        num_frames = ceildiv(num_bases_window, self.resolution)
        if not MIN_FRAMES <= num_frames:
            text = " skipping short sequence of length %d" % num_frames
            print >>sys.stderr, text
            return []

        if num_frames > max_frames:
            # XXX: should check that this is going to always work even for
            # corner cases (what corner cases?), but if it doesn't,
            # another split later on fixes it

            # split_sequences was True, so split them
            num_new_starts = ceildiv(num_frames, max_frames)

            # // means floor division
            offset = (num_frames // num_new_starts)
            new_offsets = arange(num_new_starts) * (offset * self.resolution)
            new_starts = start + new_offsets
            new_ends = append(new_starts[1:], end)

            return zip(new_starts, new_ends)

        return [[start, end]]

    def locate_windows(self, genome):
        """
        input: Genome instance, include_coords, exclude_ coords, max_frames

        sets: window_coords
        """
        exclude_coords = self.exclude_coords

        windows = []

        for chrom, starts, ends in self.generate_coords(genome):
            chr_exclude_coords = get_chrom_coords(exclude_coords, chrom)

            while True:
                try:
                    start = starts.popleft()
                except IndexError:
                    break

                end = ends.popleft()  # should not ever cause an IndexError

                new_windows = find_overlaps_exclude(start, end,
                                                    chr_exclude_coords)
                start, end = process_new_windows(new_windows, starts, ends)
                if start is None:
                    continue

                # skip or split long sequences
                new_windows = self.skip_or_split_window(start, end)
                start, end = process_new_windows(new_windows, starts, ends)
                if start is None:
                    continue

                for world in xrange(self.num_worlds):
                    windows.append(Window(world, chrom, start, end))

        self.windows = windows

    def save_window(self, float_filename, int_filename, float_data,
                    seq_data=None, supervision_data=None):
        return _save_window(float_filename, int_filename, float_data,
                            self.resolution, self.distribution, seq_data,
                            supervision_data)

    @staticmethod
    def make_filepath(dirpath, prefix, suffix):
        return dirpath / (prefix + suffix)

    def make_filepaths(self, chrom, window_index, temp=False):
        prefix_feature_tmpl = extjoin(chrom, make_prefix_fmt(MAX_WINDOWS))
        prefix = prefix_feature_tmpl % window_index

        if temp:
            prefix = "".join([prefix, self.uuid, extsep])
            dirpath = path(gettempdir())
        else:
            dirpath = self.obs_dirpath

        make_filepath_custom = partial(self.make_filepath, dirpath, prefix)

        return (make_filepath_custom(EXT_FLOAT), make_filepath_custom(EXT_INT))

    def print_filepaths(self, float_filelist, int_filelist, *args, **kwargs):
        float_filepath, int_filepath = self.make_filepaths(*args, **kwargs)
        print >>float_filelist, float_filepath
        print >>int_filelist, int_filepath

        self.float_filepaths.append(float_filepath)
        self.int_filepaths.append(int_filepath)

        return float_filepath, int_filepath

    def make_seq_cells(self, chromosome, start, end):
        if self.use_dinucleotide:
            return chromosome.seq[start:end]

    def make_continuous_cells(self, world, chromosome, start, end):
        """
        returns 2-dimensional numpy.ndarray of continuous observation
        data for specified interval. This data is untransformed

        dim 0: position
        dim 1: track
        """
        track_indexes = self.world_track_indexes[world]
        return chromosome[start:end, track_indexes]

    def make_supervision_cells(self, chrom, start, end):
        """
        chrom: str
        start: int
        end: int

        returns either None (unsupervised) or 1-dimensional
        numpy.ndarray for the region specified by chrom, start, end,
        where each cell is the transformed label of that region

        the transformation results in the cell being 0 for no supervision
        or SUPERVISION_LABEL_OFFSET (1)+the supervision label for supervision
        """
        supervision_type = self.supervision_type
        if supervision_type == SUPERVISION_UNSUPERVISED:
            return  # None

        assert supervision_type == SUPERVISION_SEMISUPERVISED

        coords_chrom = self.supervision_coords[chrom]
        labels_chrom = self.supervision_labels[chrom]

        res = zeros(end - start, dtype=DTYPE_OBS_INT)

        supercontig_coords_labels = \
            find_overlaps_include(start, end, coords_chrom, labels_chrom)

        for label_start, label_end, label_index in supercontig_coords_labels:
            # adjust so that zero means no label
            label_adjusted = label_index + SUPERVISION_LABEL_OFFSET
            res[(label_start - start):(label_end - start)] = label_adjusted

        return res

    def write(self, genome, float_filelist, int_filelist, float_tabfile):
        print_filepaths_custom = partial(self.print_filepaths,
                                         float_filelist, int_filelist,
                                         temp=self.identify)
        save_window = self.save_window

        float_tabwriter = ListWriter(float_tabfile)
        float_tabwriter.writerow(FLOAT_TAB_FIELDNAMES)

        for window_index, window in enumerate(self.windows):
            world, chrom, start, end = window
            float_filepath, int_filepath = \
                print_filepaths_custom(chrom, window_index)

            row = [float_filepath, str(window_index), chrom, str(start),
                   str(end)]
            float_tabwriter.writerow(row)
            print >>sys.stderr, " %s (%d, %d)" % (float_filepath, start, end)

            # if they don't both exist
            if not (float_filepath.exists() and int_filepath.exists()):
                chromosome = genome[chrom]
                seq_cells = self.make_seq_cells(chromosome, start, end)

                supervision_cells = \
                    self.make_supervision_cells(chrom, start, end)

                if __debug__:
                    if self.identify:
                        assert seq_cells is None and supervision_cells is None

                if not self.train:
                    # don't actually write data--it is done by task.py instead
                    continue

                continuous_cells = \
                    self.make_continuous_cells(world, chromosome, start, end)

                # data is transformed as part of _save_window()
                save_window(float_filepath, int_filepath, continuous_cells,
                            seq_cells, supervision_cells)

    def open_writable_or_dummy(self, filepath):
        if not filepath or (not self.clobber and filepath.exists()):
            return closing(StringIO())  # dummy output
        else:
            return open(filepath, "w")

    def save(self, genome):
        open_writable = self.open_writable_or_dummy

        with open_writable(self.float_filelistpath) as float_filelist:
            with open_writable(self.int_filelistpath) as int_filelist:
                with open_writable(self.float_tabfilepath) as float_tabfile:
                    self.write(genome, float_filelist, int_filelist,
                               float_tabfile)
