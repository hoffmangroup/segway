#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

"""observations.py: prepare and save GMTK observations
"""

__version__ = "$Revision$"

# Copyright 2012, 2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>
from collections import deque
from contextlib import closing
from functools import partial
from itertools import repeat
from operator import itemgetter
from os import extsep
import sys
from tempfile import gettempdir
from warnings import warn

from genomedata import Genome
from numpy import (add, any, append, arange, arcsinh, argmax, array,
                   bincount, clip, column_stack, copy, empty, full, invert,
                   isnan, maximum, mean, sum, where, zeros)
from numpy import sum as numpy_sum
from path import Path
from six import viewitems
from six.moves import map, range, StringIO, zip
from tabdelim import ListWriter

from ._util import (ceildiv, copy_attrs, DISTRIBUTION_ASINH_NORMAL,
                    DTYPE_OBS_INT, EXT_FLOAT, EXT_INT, extjoin,
                    get_chrom_coords, make_prefix_fmt, SegwayWarning,
                    SUPERVISION_LABEL_OFFSET, USE_MFSDG, Window)

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

DIM_TRACK = 1  # Dimension in numpy array for track data

PRIOR_AXIS = 0
POSITION_AXIS = 1

class VirtualEvidenceWarning(SegwayWarning):
    """
    User-supplied priors sum to greater than one.
    """


class NoData(object):
    """
    sentinel for not adding an extra field to coords, so that one can
    still use None
    """


def convert_windows(attrs, name):
    supercontig_start = attrs.start
    edges_array = getattr(attrs, name) + supercontig_start

    return edges_array.tolist()


def merge_windows(windows):
    """Takes a sorted list of start/end tuples and returns a list of tuples of
    merged together if any overlap"""

    res = []

    # Get first region
    merge_start, merge_end = windows[0]

    # For all remaining regions
    for window_start, window_end in windows[1:]:
        # If the next region does not overlap the current one under
        # consideration
        if window_start - merge_end > 0:
            # Add our current merged region to the merged list
            res.append((merge_start, merge_end))
            # Set the current region under consideration to the current window
            merge_start, merge_end = window_start, window_end
        # Otherwise the regions overlap
        else:
            if window_end > merge_end:
                merge_end = window_end

    # Append the last region under consideration to the merge list
    res.append((merge_start, merge_end))
    # Return the list of merged windows
    return res


def update_starts(starts, ends, new_starts, new_ends):
    # reversed because extend left extends deque in reversed order
    starts.extendleft(reversed(new_starts))
    ends.extendleft(reversed(new_ends))


def intersect_regions(start, end, coords, data=repeat(NoData)):
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

    for (include_start, include_end), datum in zip(coords, data):
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


def subtract_regions(start, end, exclude_coords):
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
    for index in range(remainder):
        res += inarray[index::resolution]

    # loop 1: remainder
    for index in range(remainder, resolution):
        # don't include the last element of res
        res[:-1] += inarray[index::resolution]

    return res


def get_downsampled_supervision_data_and_presence(input_array, resolution):
    """
    Downsample a 1-dimensional numpy array to a desired resolution
    by taking the mode of each resolution-sized frame.
    For example, [1 2 2 3 4 4] downsampled to resolution 3 is: [2 4]

    This function returns a tuple (where each element is an array) of
    the form: [downsampled input array], [presence of downsampled
    input array] where 'presence' is the count of each mode in its
    'window'.

    Only downsamples 1D numpy arrays.

    There is the possibility that there will be a 'remainder'
    subarray. For that case, we choose to take its mode and append
    it to the end of our array of modes.

    As a rule, we will never choose 0 (no label) unless all elements
    of the subarray are 0.

    For example, for [1, 0, 0,...,0] our 'mode' will be 1.
    But [0,0,0,0...,0] has a mode of 0.

    """
    if resolution == 1:
        # at resolution==1, the presence for all nonzero values is 1
        # and the presence for 0 is 0 (no supervision data)
        presence_array = clip(input_array, 0, 1)
        return input_array, presence_array

    # split input_array into subarrays of length resolution.
    # e.g. [1, 1, 3, 4, 5, 6] to [[1, 1, 3], [4, 5, 6] for resolution == 3
    resolution_partitioned_input_array = (
            input_array[index:index+resolution]
            for index in range(0, len(input_array), resolution)
            )

    downsampled_input_array = zeros(calc_downsampled_shape(input_array,
                                    resolution), input_array.dtype)
    presence_downsampled_input_array = zeros(calc_downsampled_shape(input_array,
                                             resolution), input_array.dtype)
    # For each input partition find its mode
    for index, input_partition in enumerate(resolution_partitioned_input_array):
        # Get the number of times each index occurs in the input partition.
        # bincount(a) returns an array where the elements are the number
        # of times each index occurs in a.
        resolution_sized_subarray_bincount = bincount(input_partition)

        # by setting the 0th value of bincount to 0, we take care
        # of two possible special cases:
        # 1) number of 0s is the same as the count of the nonzero mode:
        #   >we want to have the nonzero mode, not 0.
        #   setting count of 0 to 0 will force argmax to default to
        #   the nonzero mode.
        #
        # 2) the case of all 0s (no nonzero mode):
        #   >we want the mode to be 0.
        #   setting count of 0 to 0 will return an argmax of 0 since
        #   there are no other numbers to choose.
        resolution_sized_subarray_bincount[0] = 0

        # in the case of ties for the max count, argmax takes the
        # smallest numbered index.
        mode = argmax(resolution_sized_subarray_bincount)
        downsampled_input_array[index] = mode

        # Get the count of the mode in our input partition
        presence_downsampled_input_array[index] = \
            resolution_sized_subarray_bincount[mode]

    return downsampled_input_array, presence_downsampled_input_array


def downsample_presence_array(presence_array, resolution):
    """
    Downsamples a 1D presence array given the specified resolution.
    Sums the presence across bins given the resolution size, so
    [0, 1, 1, 1, 0, 0] for resolution == 3 becomes
    [2, 1]
    """
    return [numpy_sum(presence_array[index:index+resolution])
            for index in range(0, len(presence_array), resolution)]


def downsample_prior_array(raw_prior_array, resolution, uniform_priors):
    """
    Downsamples a 2D array of priors. Takes the average for all labels defined,
    uses a uniform prior if none are.
    """
    # split input_array into subarrays of length resolution.
    # e.g. [1,1,3,4,5,6] to [[1,1,3],[4,5,6]] for resolution == 3
    resolution_partitioned_prior_gen = (
            raw_prior_array[index:index+resolution]
            for index in range(0, len(raw_prior_array), resolution))

    # Empty array to be filled with mean values for priors for each bin
    res = empty(calc_downsampled_shape(raw_prior_array,
                                       resolution), raw_prior_array.dtype)

    # For each input partition, calculate the mean prior for each label
    # if no priors given, use a uniform prior
    for index, input_partition in enumerate(resolution_partitioned_prior_gen):
        # if no priors are defined in this partition,
        # (meaning input_partition will be entirely composed of 0s)
        # set the mean vector to be uniform
        if not input_partition.any():
            res[index] = uniform_priors
        else:
            # Remove any rows which don't contain priors
            defined_priors = input_partition[input_partition.any(POSITION_AXIS)]

            # take the mean of the given priors for each label over the
            # position axis
            mean_prior_vector = mean(defined_priors, axis=PRIOR_AXIS)

            res[index] = mean_prior_vector

    return res

def get_downsampled_virtual_evidence_data_and_presence(raw_prior_array,
                                                       resolution, num_labels):
    """
    Downsample a 2-dimensional array of label probabilities to a
    desired resolution by taking the average for all labels
    over the positions with priors.

    If no priors are specified then the average prior is uniform.

    This function returns a tuple (where each element is an array) of
    the form: [downsampled input array], [presence of downsampled
    input array] where 'presence' is the count of evidence-specified positions
    in each subpartition.

    Downsampled input array is an array of arrays where each element
    is the vector of priors for the given downsampled position.

    There is the possibility that there will be a 'remainder'
    subarray. For that case, we choose to take its average (same as above)
    and append it to the end of our array of downsampled priors.

    For example, the 3-position frame

    [[0.5, 0.2, 0.3],
     [0.3, 0.4, 0.3],
     [0.0, 0.0, 0.0]]

    downsampled to resolution 3 is [0.4, 0.3, 0.3]
    since for label 0 we take mean(0.5, 0.3) = 0.4
          for label 1 we take mean(0.2, 0.4) = 0.3
          for label 2 we take mean(0.3, 0.3) = 0.3
    """

    uniform_priors = array([1.0/num_labels] * num_labels)

    # take the presence to be 1 at every position the user has defined
    # any priors and 0 otherwise
    presence_array = raw_prior_array.any(axis=POSITION_AXIS).astype(DTYPE_OBS_INT)

    if resolution == 1:
        # our "downsampled" prior array at resolution 1 is just the
        # vector of priors defined by the user at every position
        # with uniform priors filled in at all other positions
        prior_array = zeros((len(raw_prior_array), num_labels))
        for prior_list_index, prior_vector in enumerate(raw_prior_array):
            if prior_vector.sum() == 0:
                prior_array[prior_list_index] = uniform_priors
            else:
                prior_array[prior_list_index] = prior_vector

        return prior_array, presence_array

    downsampled_prior_array = downsample_prior_array(raw_prior_array,
                                                     resolution,
                                                     uniform_priors)
    downsampled_presence_array = downsample_presence_array(presence_array,
                                                           resolution)


    return downsampled_prior_array, downsampled_presence_array


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


def make_supervision_cells(supervision_coords, supervision_labels, start, end):
    """
    supervision_coords: list of tuples (start, end)
    supervision_labels: list of ints (label as number)
    start: int
    end: int

    returns a 1-dimensional numpy.ndarray for the region specified by
    superivion_coords where each cell is the transformed label specified by
    supervision_labels for that region

    the transformation results in the cell being 0 for no supervision
    or SUPERVISION_LABEL_OFFSET (1)+the supervision label for supervision
    """

    res = zeros(end - start, dtype=DTYPE_OBS_INT)

    # Get supervision regions that overlap with the start and end coords
    supercontig_coords_labels = \
        intersect_regions(start, end, supervision_coords,
                          supervision_labels)

    for label_start, label_end, label_index in supercontig_coords_labels:
        # adjust so that zero means no label
        label_adjusted = label_index + SUPERVISION_LABEL_OFFSET
        res[(label_start - start):(label_end - start)] = label_adjusted

    return res


def fill_virtual_evidence_cells(prior_input_array, num_labels):
    """
    For genomic positions which have at least one, but not all priors specified,
    this function will apply a uniform prior to all remaining labels.
    Indexes where no prior was ever specified will remain zero for downsampling

    Example:
    INPUT:
    [[0.5, 0.2, None, None, None],
     [None, 0.4, None, None, None],
     [None, None, None, None, None]]

    OUTPUT: 
    [[0.5, 0.2, 0.10, 0.10, 0.10],
     [0.15, 0.4, 0.15, 0.15, 0.15],
     [0.0 ... 0.0]]
    """

    prior_array = zeros((len(prior_input_array), num_labels))
    for index, prior_input in enumerate(prior_input_array):
        # Only priors which were specified in the input file will be set
        # Unset labels will still be none
        num_prior_labels = numpy_sum(prior_input != None)
        if num_prior_labels:
            prior_list_values = list(filter(None, prior_input))

            # Check if priors should be treated as ratios or percentages
            if numpy_sum(prior_list_values) < 1:
                remaining_probability = 1 - numpy_sum(prior_list_values)
            else:
                remaining_probability = 0

            # divide remaining probability uniformly amongst the remaining labels
            prior_input[prior_input == None] = (remaining_probability /
                                                (num_labels-num_prior_labels))

            prior_array[index] = prior_input

    return prior_array

def check_is_close(prior_sums, reference):
    return all((prior_sums - reference) <= 0)

def make_virtual_evidence_cells(coords, priors,
                                start, end, num_labels):
    """
    coords: list of tuples (start, end)
    priors: list of dictionaries of the format {label: prior}
    start: int
    end: int

    returns a 2-dimensional numpy.ndarray for the section of the array bounded
    by start and end where each cell is the prior data for each label of each
    region inside the section.
    """
    res = full((num_labels, (end-start)), None)

    # Get supervision regions that overlap with the start and end coords
    supercontig_coords_labels = \
        intersect_regions(start, end, coords, priors)

    for label_start, label_end, prior_dict in supercontig_coords_labels:
        label = list(prior_dict.keys())[0]
        # copy data to all positions
        if res[label][(label_start - start):(label_end - start)].any():
            raise ValueError("VE label {} overlaps in coordinates {}-{}".format(label,
                             label_start, label_end))
        res[label][(label_start - start):(label_end - start)] = \
            list(prior_dict.values()) * ((label_end - start)-(label_start - start))

    # For coords which had at least one label supplied, fill remaining labels
    # with uniform remaining probability
    # Transpose the priors so that it has positions as rows and labels as columns
    res = fill_virtual_evidence_cells(res.transpose(), num_labels)

    if any(res < 0):
        raise ValueError("Priors may not be negative")

    is_close = check_is_close(res.sum(axis=POSITION_AXIS), 1.0)
    if not is_close:
        close_indexes = where(res.sum(axis=POSITION_AXIS) > 1.0)[0]
        warn("Prior labels sum to greater than one in on genomic indexes {}".format(close_indexes + start),
             VirtualEvidenceWarning)

    return res


def make_continuous_cells(track_indexes, genomedata_names,
                          chromosome_name, start, end):
    """
    returns 2-dimensional numpy.ndarray of continuous observation
    data for specified interval. This data is untransformed

    dim 0: position
    dim 1: track
    """

    continuous_cells = None

    # For every track in each genomedata archive
    zipper = zip(track_indexes, genomedata_names)
    for track_index, genomedata_name in zipper:
        with Genome(genomedata_name) as genome:
            chromosome = genome[chromosome_name]
            # If we haven't started creating the continous cells
            if continuous_cells is None:
                # Copy the first track into our continous cells
                continuous_cells = copy(chromosome[start:end,
                                        [track_index]])
            else:
                # Otherwise append the track to our continuous cells
                continuous_cells = append(continuous_cells,
                                          chromosome[start:end,
                                                     [track_index]],
                                          DIM_TRACK)

    return continuous_cells


def _save_window(float_filename_or_file, int_filename_or_file, 
                 float_data, resolution, distribution, seq_data=None,
                 supervision_data=None, virtual_evidence_data=None,
                 virtual_evidence_filename_or_file=None, num_labels=None):
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

        float_data.tofile(float_filename_or_file)

    if seq_data is not None:
        assert resolution == 1  # not implemented yet
        int_blocks.append(make_dinucleotide_int_data(seq_data))

    if supervision_data is not None:
        supervision_data, presence_supervision_data = \
            get_downsampled_supervision_data_and_presence(supervision_data,
                                                          resolution)

        int_blocks.append(supervision_data)
        int_blocks.append(presence_supervision_data)

    if virtual_evidence_data is not None:
        virtual_evidence_data_array, presence_virtual_evidence_data = \
            get_downsampled_virtual_evidence_data_and_presence(
                virtual_evidence_data,
                resolution,
                num_labels)
        int_blocks.append(presence_virtual_evidence_data)

        # separately save VE priors CPT in a temporary file
        # done using write since ndarray.tofile does not format correctly
        # Produces error: observation file 0 '/tmp/segway.qvHHEnnQV4/ve.gradckoq.list' segment 0: couldn't read 0'th item in frame 0
        for prior in virtual_evidence_data_array:
            virtual_evidence_filename_or_file.write(
                ' '.join(['{}'.format(prob) for prob in prior]) + '\n')

    if int_blocks:
        int_data = column_stack(int_blocks).astype(dtype=DTYPE_OBS_INT)
    else:
        int_data = array([], dtype=DTYPE_OBS_INT)

    int_data.tofile(int_filename_or_file)

def add_starts_ends(new_windows, starts, ends):
    """
    If there are no new windows, return None, None

    If there is more than one new window, modify the
    starts and ends in place and return None, None

    If there is only one new window, return a tuple of
    (start, end) for that window.

    """
    if not new_windows:  # nothing left
        return None, None
    elif len(new_windows) > 1:
        new_starts, new_ends = zip(*new_windows)
        # reversed because extend left extends deque in reversed order
        starts.extendleft(reversed(new_starts))
        ends.extendleft(reversed(new_ends))
        return None, None

    return new_windows[0]

def generate_coords_from_dict(coords_dict):
    # use a deque to allow fast insertion/removal at the
    # beginning and end of the sequence
    for chrom, coords_list in viewitems(coords_dict):
        starts, ends = map(deque, zip(*coords_list))
        yield chrom, starts, ends

class Observations(object):
    copy_attrs = ["include_coords", "exclude_coords", "max_frames",
                  "float_filelistpath", "int_filelistpath",
                  "validation_float_filelistpath",
                  "validation_int_filelistpath",
                  "float_tabfilepath", "validation_obs_dirpath",
                  "obs_dirpath", "uuid", "resolution",
                  "distribution", "train", "identify", "random_state",
                  "supervision_type",
                  "supervision_coords", "supervision_labels",
                  "use_dinucleotide", "world_track_indexes",
                  "world_genomedata_names", "clobber",
                  "num_worlds", "validation_fraction", "validate",
                  "validation_coords", "virtual_evidence"]

    def __init__(self, runner):
        copy_attrs(runner, self, self.copy_attrs)

        self.float_filepaths = []
        self.int_filepaths = []

        self.validation_float_filepaths = []
        self.validation_int_filepaths = []

        self.validation_windows = []

    def generate_coords_all(self, genomes):
        """Generates a tuple of (chromosome, starts, ends) for each chromosome
        across all genomes where the starts and ends are deques"""

        # Get all chromosome names across all genomes
        chromosome_names = set()
        for genome in genomes:
            chromosome_names = chromosome_names.union([chromosome.name for
                                                       chromosome in genome])

        # For each chromsome (in sorted order)
        for chromosome_name in sorted(list(chromosome_names)):
            chromosome_windows = []  # list of (start, end) tuples

            # Get all genome-mapped regions across all genomes
            for genome in genomes:
                chromosome = genome[chromosome_name]

                # For each supercontig
                for supercontig, continuous in chromosome.itercontinuous():
                    # If the the supercontig is not empty
                    if continuous is not None:
                        attrs = supercontig.attrs

                        starts = convert_windows(attrs, "chunk_starts")
                        ends = convert_windows(attrs, "chunk_ends")

                        chromosome_windows.extend(zip(starts, ends))

            # If at least 1 window exists
            if chromosome_windows:
                # Sort regions (by start)
                chromosome_windows.sort(key=itemgetter(0))
                # Merge regions
                merged_chromsome_windows = merge_windows(chromosome_windows)
                # Convert start and end regions to deques
                starts, ends = map(deque, zip(*merged_chromsome_windows))
                # Yield the chromsome name and start/end region deques
                yield chromosome_name, starts, ends

    def generate_coords(self, genomes):
        """
        returns iterable of included coords, either explicitly
        specified, or all

          each item: tuple of (chrom, starts, ends)
          starts and ends are deques
        """
        if self.include_coords:
            return generate_coords_from_dict(self.include_coords)
        else:
            return self.generate_coords_all(genomes)

    def skip_or_split_window(self, start, end):
        """
        skip short windows or skip long windows
        """
        max_frames = self.max_frames
        num_bases_window = end - start
        num_frames = ceildiv(num_bases_window, self.resolution)
        if not MIN_FRAMES <= num_frames:
            text = " skipping short sequence of length %d" % num_frames
            print(text, file=sys.stderr)
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

            return list(zip(new_starts, new_ends))

        return [[start, end]]

    def create_validation_windows_from_fraction(self, windows):
        """
        Takes a subset of the training windows, using the specified
        validation fraction, to be used as the validation windows.

        Note: modifies the input parameter 'windows' and uses this
        as the training windows.

        Returns training windows, validation windows
        """
        validation_windows = []
        total_bases = sum(len(window) for window in windows)
        cur_bases = 0
        self.random_state.shuffle(windows)

        # Remove windows from training set and add them to validation set
        # until total number of bases chosen is at least
        # as large as that required by the validation fraction
        while (float(cur_bases) / total_bases) < self.validation_fraction:
            window = windows.pop()
            validation_windows.append(window)
            cur_bases += len(window)

        return windows, validation_windows

    def create_validation_windows_from_coords(self, exclude_coords):
        """
        Creates set of validation windows using the specified
        validation coordinates.

        Returns validation windows
        """
        validation_windows = []
        for chrom, starts, ends in generate_coords_from_dict(self.validation_coords):
            chr_exclude_coords = get_chrom_coords(exclude_coords, chrom)

            while len(starts) > 0:
                start = starts.popleft()
                end = ends.popleft()  # should not ever cause an IndexError

                new_validation_windows = \
                    subtract_regions(start, end, chr_exclude_coords)
                subtracted_start, subtracted_end = \
                    add_starts_ends(new_validation_windows,
                                        starts,
                                        ends)
                if subtracted_start:
                    # skip or split long sequences
                    new_validation_windows = \
                        self.skip_or_split_window(subtracted_start,
                                                  subtracted_end)
                    split_start, split_end = \
                        add_starts_ends(new_validation_windows,
                                            starts,
                                            ends)
                    if split_start:
                        for world in range(self.num_worlds):
                            validation_windows.append(Window(
                                world, chrom, split_start, split_end)
                                )

        return validation_windows

    def locate_windows(self, genomes):
        """
        input: Genome instances, include_coords, exclude_ coords, max_frames
        validation fraction/validation coords (if validation)

        sets: window_coords, validation_coords (if validation)
        """
        exclude_coords = self.exclude_coords

        windows = []

        for chrom, starts, ends in self.generate_coords(genomes):
            chr_exclude_coords = get_chrom_coords(exclude_coords, chrom)

            while True:
                try:
                    start = starts.popleft()
                except IndexError:
                    break

                end = ends.popleft()  # should not ever cause an IndexError

                new_windows = subtract_regions(start, end, chr_exclude_coords)
                start, end = add_starts_ends(new_windows, starts, ends)
                if start is None:
                    continue

                # skip or split long sequences
                new_windows = self.skip_or_split_window(start, end)
                start, end = add_starts_ends(new_windows, starts, ends)
                if start is None:
                    continue

                for world in range(self.num_worlds):
                    windows.append(Window(world, chrom, start, end))

        if self.validation_fraction:
            windows, self.validation_windows = \
                self.create_validation_windows_from_fraction(windows)
        elif self.validation_coords:
            self.validation_windows = \
                self.create_validation_windows_from_coords(exclude_coords)

        if not windows:
            raise ValueError("Set of training windows is empty")

        # remaining windows can be passed into training now
        self.windows = windows

    def save_window(self, float_filename, int_filename, float_data,
                    seq_data=None, supervision_data=None):
        return _save_window(float_filename, int_filename, float_data,
                            self.resolution, self.distribution, seq_data,
                            supervision_data)

    @staticmethod
    def make_filepath(dirpath, prefix, suffix):
        return dirpath / (prefix + suffix)

    def make_validation_filepaths(self, chrom, window_index):
        prefix_feature_tmpl = extjoin(chrom, make_prefix_fmt(MAX_WINDOWS))
        prefix = prefix_feature_tmpl % window_index

        dirpath = self.validation_obs_dirpath

        make_filepath_custom = partial(self.make_filepath, dirpath, prefix)

        return (make_filepath_custom(EXT_FLOAT), make_filepath_custom(EXT_INT))

    def make_filepaths(self, chrom, window_index, temp=False):
        prefix_feature_tmpl = extjoin(chrom, make_prefix_fmt(MAX_WINDOWS))
        prefix = prefix_feature_tmpl % window_index

        if temp:
            prefix = "".join([prefix, self.uuid, extsep])
            dirpath = Path(gettempdir())
        else:
            dirpath = self.obs_dirpath

        make_filepath_custom = partial(self.make_filepath, dirpath, prefix)

        return (make_filepath_custom(EXT_FLOAT), make_filepath_custom(EXT_INT))

    def print_filepaths(self, float_filelist, int_filelist, *args, **kwargs):
        float_filepath, int_filepath = self.make_filepaths(*args, **kwargs)
        print(float_filepath, file=float_filelist)
        print(int_filepath, file=int_filelist)

        self.float_filepaths.append(float_filepath)
        self.int_filepaths.append(int_filepath)

        return float_filepath, int_filepath

    def create_filepaths(self, temp=False):
        """ Creates a list of observations full filepaths for each window.
        temp is a flag to determine whether or not the filepaths will be
        given a temporary directory """

        for window_index, window in enumerate(self.windows):
            float_filepath, int_filepath = self.make_filepaths(
                                            window.chrom, window_index,
                                            temp)
            self.float_filepaths.append(float_filepath)
            self.int_filepaths.append(int_filepath)

    def create_validation_filepaths(self):
        """ Creates a list of observations full filepaths for each
        validation window."""

        for window_index, window in enumerate(self.validation_windows):
            float_filepath, int_filepath = self.make_validation_filepaths(
                                            window.chrom, window_index)
            self.validation_float_filepaths.append(float_filepath)
            self.validation_int_filepaths.append(int_filepath)

    def make_seq_cells(self, chromosome, start, end):
        if self.use_dinucleotide:
            return chromosome.seq[start:end]

    # XXX: Dead code. Observations.save (which calls this function) no longer
    # called. May be used when --observations is re enabled for caching
    # observation files
    # XXX: Determine if posterior will also use a temporary file path as well
    def write_tab_file(self, float_filelist, int_filelist, float_tabfile):
        print_filepaths_custom = partial(self.print_filepaths,
                                         float_filelist, int_filelist,
                                         temp=(self.identify or self.train))

        float_tabwriter = ListWriter(float_tabfile)
        float_tabwriter.writerow(FLOAT_TAB_FIELDNAMES)

        for window_index, window in enumerate(self.windows):
            world, chrom, start, end = window
            float_filepath, int_filepath = \
                print_filepaths_custom(chrom, window_index)

            row = [float_filepath, str(window_index), chrom, str(start),
                   str(end)]
            float_tabwriter.writerow(row)
            print(" %s (%d, %d)" % (float_filepath, start, end), file=sys.stderr)

    def open_writable_or_dummy(self, filepath):
        if not filepath or (not self.clobber and filepath.exists()):
            return closing(StringIO())  # dummy output
        else:
            return open(filepath, "w")

    # XXX: Dead code. Observations.save no longer called. May be used when
    # --observations is re enabled for caching observation files
    def save(self):
        open_writable = self.open_writable_or_dummy

        with open_writable(self.float_filelistpath) as float_filelist:
            with open_writable(self.int_filelistpath) as int_filelist:
                with open_writable(self.float_tabfilepath) as float_tabfile:
                    self.write_tab_file(float_filelist, int_filelist, float_tabfile)
