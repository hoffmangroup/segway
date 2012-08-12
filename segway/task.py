#!/usr/bin/env python
from __future__ import division, with_statement

"""
task: wraps a GMTK subtask to reduce size of output
"""

__version__ = "$Revision$"

# Copyright 2009-2012 Michael M. Hoffman <mmh1@washington.edu>

from errno import ENOENT
from os import extsep, fdopen
import os
import re
import sys
import uuid
import subprocess # XXX
import time # XXX
from tempfile import gettempdir, mkstemp

from genomedata import Genome
from numpy import array, empty, where, diff, r_, zeros
import numpy as np # XXX
from path import path

from .observations import _save_window
from ._util import (BED_SCORE, BED_STRAND, ceildiv, DTYPE_IDENTIFY, EXT_FLOAT,
                    EXT_INT, EXT_LIST, fill_array, find_segment_starts,
                    get_label_color,
                    POSTERIOR_PROG, POSTERIOR_SCALE_FACTOR, read_posterior,
                    VITERBI_PROG)

MSG_SUCCESS = "____ PROGRAM ENDED SUCCESSFULLY WITH STATUS 0 AT"

SCORE_MIN = 100
SCORE_MAX = 1000

SEG_INVALID = -1

TEMP_DIRPATH = path(gettempdir())

EXT_OPTIONS = {}
EXT_OPTIONS[EXT_FLOAT] = "-of1" # duplicative of run.py
EXT_OPTIONS[EXT_INT] = "-of2"

def make_track_indexes(text):
    return array(map(int, text.split(",")))

re_seg = re.compile(r"^seg\((\d+)\)=(\d+)$")
def parse_viterbi(lines, do_reverse=False):
    """
    returns: numpy.ndarray of size (num_frames,), type DTYPE_IDENTIFY
    """
    lines = iter(lines)

    # Segment 0, after Island[...]
    assert lines.next().startswith("Segment ")

    # ========
    assert lines.next().startswith("========")

    # Segment 0, number of frames = 1001, viteri-score = -6998.363710
    line = lines.next()
    assert line.startswith("Segment ")

    num_frames_text = line.split(", ")[1].partition(" = ")
    assert num_frames_text[0] == "number of frames"
    assert num_frames_text[1] == " = "

    num_frames = int(num_frames_text[2])

    # Printing random variables from (P,C,E)=(1,999,0) partitions
    line = lines.next()
    assert line.startswith("Printing random variables from (P,C,E)")

    # sentinel value
    res = fill_array(SEG_INVALID, num_frames, DTYPE_IDENTIFY)

    for line in lines:
        # Ptn-0 P': seg(0)=24,seg(1)=24
        if line.startswith(MSG_SUCCESS):
            assert (res != SEG_INVALID).all()
            return res

        assert line.startswith("Ptn-")

        values = line.rpartition(": ")[2]

        for pair in values.split(","):
            match = re_seg.match(pair)
            if not match:
                continue

            index = int(match.group(1))
            if do_reverse:
                index = -1 - index # -1, -2, -3, etc.

            val = int(match.group(2))

            res[index] = val

    # shouldn't get to this point
    raise ValueError("%s did not complete successfully" % VITERBI_PROG.prog)

# num_cols is for use by genomedata_count_condition
# XXX: should move this function somewhere else
def write_bed(outfile, start_pos, labels, coord, resolution, num_labels,
              num_cols=None):
    """
    start_pos is an array
    """
    (chrom, region_start, region_end) = coord

    start_pos = region_start + (start_pos * resolution)

    # correct last position which may not be a full sample beyond
    if __debug__:
        remainder = (region_end - region_start) % resolution
        if remainder == 0:
            remainder = resolution
        assert region_end == start_pos[-1] - resolution + remainder
    start_pos[-1] = region_end

    # score_step = (SCORE_MAX - SCORE_MIN) / (num_labels - 1)

    zipper = zip(start_pos[:-1], start_pos[1:], labels)

    # this is easily concatenated since it has no context
    for seg_start, seg_end, seg_label in zipper:
        name = str(seg_label)

        chrom_start = str(seg_start)
        chrom_end = str(seg_end)
        item_rgb = get_label_color(seg_label)

        row = [chrom, chrom_start, chrom_end, name, BED_SCORE, BED_STRAND,
               chrom_start, chrom_end, item_rgb][:num_cols]

        print >>outfile, "\t".join(row)

    # assert that the whole region is mapped
    # seg_end here means the last seg_end in the loop
    assert seg_end == region_end

def save_bed(outfilename, *args, **kwargs):
    with open(outfilename, "w") as outfile:
        write_bed(outfile, *args, **kwargs)

def read_posterior_save_bed(coord, resolution, do_reverse, outfilename_tmpl, num_labels,
                            infile):
    print >>sys.stderr, "got here 10"
    sys.stderr.flush()
    if do_reverse:
        raise NotImplementedError

    print >>sys.stderr, "got here 11"
    sys.stderr.flush()
    (chrom, start, end) = coord
    num_frames = ceildiv(end - start, resolution)
    probs = read_posterior(infile, num_frames, num_labels)
    # XXX
    probs_rounded = empty(probs.shape, float)
    #probs_rounded = empty(probs.shape, int)
    print >>sys.stderr, "got here 12"
    sys.stderr.flush()

    outfilenames = []
    for label_index in xrange(num_labels):
        outfilenames.append(outfilename_tmpl % label_index)
    print >>sys.stderr, "got here 13"
    sys.stderr.flush()

    # scale, round, and cast to int
    # XXX
    #(probs * POSTERIOR_SCALE_FACTOR).round(out = probs_rounded)
    (probs * POSTERIOR_SCALE_FACTOR).round(decimals=100, out = probs_rounded)
    print >>sys.stderr, "got here 14"
    sys.stderr.flush()

    # print array columns as text to each outfile
    zipper = zip(outfilenames, probs_rounded.T, xrange(num_labels))
    for outfilename, probs_rounded_label, label_index in zipper:
        # run-length encoding on the probs_rounded_label
        sys.stderr.flush()

        outfile = open(outfilename, "w")
        pos, = where(diff(probs_rounded_label) != 0)
        pos = r_[start, pos[:]*resolution+start+resolution, end]
        sys.stderr.flush()

        for bed_start, bed_end in zip(pos[:-1], pos[1:]):
            chrom_start = str(bed_start)
            chrom_end = str(bed_end)
            value = str(probs_rounded_label[(bed_start-start)/resolution])
            sys.stderr.flush()

            row = [chrom, chrom_start, chrom_end, value]
            print >>outfile, "\t".join(row)

def load_posterior_save_bed(coord, resolution, do_reverse, outfilename, num_labels,
                            infilename):
    with open(infilename) as infile:
        read_posterior_save_bed(coord, resolution, do_reverse, outfilename,
                                int(num_labels), infile)

def parse_viterbi_save_bed(coord, resolution, do_reverse, viterbi_lines, bed_filename, num_labels):
    data = parse_viterbi(viterbi_lines, do_reverse)

    start_pos, labels = find_segment_starts(data)

    save_bed(bed_filename, start_pos, labels, coord, resolution, int(num_labels))

def load_viterbi_save_bed(coord, resolution, do_reverse, outfilename, num_labels, infilename):
    with open(infilename) as infile:
        lines = infile.readlines()

    return parse_viterbi_save_bed(coord, resolution, do_reverse, lines, outfilename,
                                  num_labels)

def replace_args_filelistname(args, temp_filepaths, ext):
    """
    replace the filelistnames in arguments with temporary filenames
    """
    fd, filelistname = mkstemp(suffix=extsep + EXT_LIST, prefix=ext + extsep)
    filelistpath = path(filelistname)

    # side-effect on args, temp_filepaths
    option = EXT_OPTIONS[ext]
    try:
        args[args.index(option) + 1] = filelistname
    except ValueError:
        pass # not going to add this filename to the command line
    temp_filepaths.append(filelistpath)

    return fd

def print_to_fd(fd, line):
    with fdopen(fd, "w") as outfile:
        print >>outfile, line

def run_posterior_save_bed(coord, resolution, do_reverse, outfilename, num_labels,
                           genomedataname, float_filename, int_filename,
                           distribution, track_indexes_text, *args):
    print >>sys.stderr, "got here 1"
    sys.stderr.flush()

    # XXX: this whole function is duplicative of run_viterbi_save_bed and needs to be reduced
    # convert from tuple
    args = list(args)

    # a 2,000,000-frame output file is only 84 MiB so it is okay to
    # read the whole thing into memory
    (chrom, start, end) = coord
    track_indexes = make_track_indexes(track_indexes_text)
    print >>sys.stderr, "got here 2"
    sys.stderr.flush()

    float_filepath = TEMP_DIRPATH / float_filename
    int_filepath = TEMP_DIRPATH / int_filename
    temp_filepaths = [float_filepath, int_filepath]
    print >>sys.stderr, "got here 3"
    sys.stderr.flush()

    # XXX: should do something to ensure of1 matches with int, of2 with float
    float_filelistfd = replace_args_filelistname(args, temp_filepaths,
                                                 EXT_FLOAT)
    int_filelistfd = replace_args_filelistname(args, temp_filepaths, EXT_INT)

    print >>sys.stderr, "got here 4"
    sys.stderr.flush()
    with Genome(genomedataname) as genome:
        continuous_cells = genome[chrom][start:end, track_indexes]

    print >>sys.stderr, "got here 5"
    sys.stderr.flush()
    try:
        print >>sys.stderr, "got here 6"
        sys.stderr.flush()
        print_to_fd(float_filelistfd, float_filename)
        print_to_fd(int_filelistfd, int_filename)

        _save_window(float_filename, int_filename, continuous_cells,
                     resolution, distribution)
        print >>sys.stderr, "got here 6.1"
        sys.stderr.flush()

        # XXXopt: does this actually free the memory? or do we need to
        # use a subprocess to do the loading?

        # remove from memory
        del continuous_cells

        print >>sys.stderr, "starting GMTK..."
        sys.stderr.flush()
        output = POSTERIOR_PROG.getoutput(*args)
        print >>sys.stderr, "done with GMTK."
        sys.stderr.flush()
    except:
        raise
    finally:
        print >>sys.stderr, "got here 7"
        sys.stderr.flush()
        for filepath in temp_filepaths:
            # don't raise a nested exception if the file was never created
            try:
                filepath.remove()
            except OSError, err:
                if err.errno == ENOENT:
                    pass

    print >>sys.stderr, "got here 8"
    sys.stderr.flush()
    lines = output.splitlines()
    print >>sys.stderr, "got here 9"
    sys.stderr.flush()
    return read_posterior_save_bed(coord, resolution, do_reverse, outfilename,
                                   int(num_labels), lines)


def run_viterbi_save_bed(coord, resolution, do_reverse, outfilename, num_labels,
                         genomedataname, float_filename, int_filename,
                         distribution, track_indexes_text, *args):
    # convert from tuple
    args = list(args)
    # a 2,000,000-frame output file is only 84 MiB so it is okay to
    # read the whole thing into memory

    (chrom, start, end) = coord

    track_indexes = make_track_indexes(track_indexes_text)

    float_filepath = TEMP_DIRPATH / float_filename
    int_filepath = TEMP_DIRPATH / int_filename

    temp_filepaths = [float_filepath, int_filepath]

    # XXX: should do something to ensure of1 matches with int, of2 with float
    int_filelistfd = replace_args_filelistname(args, temp_filepaths, EXT_INT)
    float_filelistfd = replace_args_filelistname(args, temp_filepaths,
                                                   EXT_FLOAT)

    with Genome(genomedataname) as genome:
        continuous_cells = genome[chrom][start:end, track_indexes]

    try:
        print_to_fd(float_filelistfd, float_filename)
        print_to_fd(int_filelistfd, int_filename)

        print >>sys.stderr, "got here 200"
        _save_window(float_filename, int_filename, continuous_cells,
                     resolution, distribution)
        print >>sys.stderr, "got here 201"

        # XXXopt: does this work? or do we need to use a subprocess to
        # do the loading?
        # remove from memory
        print >>sys.stderr, "got here 202"
        del continuous_cells

        print >>sys.stderr, "running viterbi program..."
        output = VITERBI_PROG.getoutput(*args)
        print >>sys.stderr, "done with viterbi program."
    finally:
        for filepath in temp_filepaths:
            # don't raise a nested exception if the file was never created
            try:
                filepath.remove()
                pass
            except OSError, err:
                if err.errno == ENOENT:
                    pass

    lines = output.splitlines()
    print >>sys.stderr, "lines:\n", lines

    return parse_viterbi_save_bed(coord, resolution, do_reverse, lines, outfilename,
                                  num_labels)

TASKS = {("run", "viterbi"): run_viterbi_save_bed,
         ("load", "viterbi"): load_viterbi_save_bed,
         ("run", "posterior"): run_posterior_save_bed,
         ("load", "posterior"): load_posterior_save_bed}

def task(verb, kind, outfilename, chrom, start, end, resolution, reverse, *args):
    print >>sys.stderr, "got here 102"
    start = int(start)
    end = int(end)
    resolution = int(resolution)
    reverse = int(reverse)

    print >>sys.stderr, "running task(verb=%s, kind=%s, outfilename=%s, chrom=%s, start=%s, end=%s, resolution=%s, reverse=%s, args=%s)" % (verb, kind, outfilename, chrom, start, end, resolution, reverse, args)
    sys.stderr.flush()

    print >>sys.stderr, "got here 103"
    return TASKS[verb, kind]((chrom, start, end), resolution, reverse, outfilename, *args)
    print >>sys.stderr, "done with TASK."
    sys.stderr.flush()

def main(args=sys.argv[1:]):
    print >>sys.stderr, "=============== starting segway-task ===================="

    if len(args) < 7:
        print >>sys.stderr, \
            "args: VERB KIND OUTFILE CHROM START END RESOLUTION REVERSE [ARGS...]"
        sys.exit(2)

    print >>sys.stderr, "args:", args

    # XXX This code fixes the really strange nondeterministic
    # segfault bug.  I have no idea why it's necessary
    args = map(lambda s: s.replace("SENTINEL_PERCENT_SIGN", "%s"), args)

    return task(*args)

if __name__ == "__main__":
    print >>sys.stderr, "got here (name __main__)"

    print >>sys.stderr, "got here 101"
    sys.exit(main())
