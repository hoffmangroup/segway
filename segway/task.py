#!/usr/bin/env python
from __future__ import division, with_statement

"""
task: wraps a GMTK subtask to reduce size of output
"""

# Copyright 2009-2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from ast import literal_eval
from errno import ENOENT
from os import extsep, fdopen, EX_TEMPFAIL
import re
import sys
from tempfile import gettempdir, mkstemp

from genomedata import Genome
from numpy import argmax, array, empty, where, diff, r_, zeros
import optbuild
from path import path

from .observations import (make_continuous_cells, make_supervision_cells,
                           _save_window)
from ._util import (BED_SCORE, BED_STRAND, ceildiv, DTYPE_IDENTIFY, EXT_FLOAT,
                    EXT_INT, EXT_LIST, extract_superlabel, fill_array,
                    find_segment_starts, get_label_color, TRAIN_PROG,
                    POSTERIOR_PROG, POSTERIOR_SCALE_FACTOR, read_posterior,
                    VITERBI_PROG)

MSG_SUCCESS = "____ PROGRAM ENDED SUCCESSFULLY WITH STATUS 0 AT"

SCORE_MIN = 100
SCORE_MAX = 1000

SEG_INVALID = -1

TEMP_DIRPATH = path(gettempdir())

EXT_OPTIONS = {}
EXT_OPTIONS[EXT_FLOAT] = "-of1"  # duplicative of run.py
EXT_OPTIONS[EXT_INT] = "-of2"

USAGE = "args: VERB KIND OUTFILE CHROM START END RESOLUTION REVERSE [ARGS...]"

# Dummy observation filename required for gmtk EM bundling. Has no effect but a
# name is necessary.
DUMMY_OBSERVATION_FILENAME = "/dev/zero"

GMTK_TRRNG_OPTION_STRING = "-trrng"  # Range to train over segment file


def make_track_indexes(text):
    return array(map(int, text.split(",")))


def divide_posterior_array(posterior_code, num_frames, num_sublabels):
    """
    takes a one-dimensional array whose values are integers of the form
    label * num_sublabels + sublabel
    and creates a two-dimensional array whose columns contain the label
    and the sublabel in separate values. This is a convenience function to
    provide the find_segment_starts() function with data in the same format
    as during the viterbi task.
    """
    res = zeros((2, num_frames), DTYPE_IDENTIFY)
    for frame_index in xrange(num_frames):
        total_label = posterior_code[frame_index]
        label, sublabel = divmod(total_label, num_sublabels)
        res[:, frame_index] = array([label, sublabel])
    return res


def parse_viterbi(lines, do_reverse=False, output_label="seg"):
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
    assert line.startswith("Printing random variables from")
    seg_dict = {'seg': 0, 'subseg': 1}
    # if output_label == "subseg" or "full", need to catch
    # subseg output
    if output_label != "seg":
        re_seg = re.compile(r"^(seg|subseg)\((\d+)\)=(\d+)$")
    else:
        re_seg = re.compile(r"^(seg)\((\d+)\)=(\d+)$")
    # sentinel value
    res = fill_array(SEG_INVALID, (2, num_frames), DTYPE_IDENTIFY)
    for line in lines:
        # Ptn-0 P': seg(0)=24,seg(1)=24
        if line.startswith(MSG_SUCCESS):
            assert (res[0] != SEG_INVALID).all()
            # if output_label == "subseg" or "full",
            # res will have 2 rows
            if output_label != "seg":
                assert (res[1] != SEG_INVALID).all()
                return res
            else:
                return res[0]

        assert line.startswith("Ptn-")

        values = line.rpartition(": ")[2]

        for pair in values.split(","):
            match = re_seg.match(pair)
            if not match:
                continue

            index = int(match.group(2))
            if do_reverse:
                index = -1 - index  # -1, -2, -3, etc.

            val = int(match.group(3))
            seg_index = seg_dict[match.group(1)]
            res[seg_index][index] = val

    # shouldn't get to this point
    raise ValueError("%s did not complete successfully" % VITERBI_PROG.prog)


# num_cols is for use by genomedata_count_condition
# XXX: should move this function somewhere else

def write_bed(outfile, start_pos, labels, coord, resolution, num_labels,
              num_cols=None, num_sublabels=None, sublabels=None):
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
    label_colors = [get_label_color(extract_superlabel(seg_label)) for
                    seg_label in labels]

    zipper = zip(start_pos[:-1], start_pos[1:], labels, label_colors)

    # this is easily concatenated since it has no context
    for seg_start, seg_end, seg_label, label_color in zipper:
        name = str(seg_label)

        chrom_start = str(seg_start)
        chrom_end = str(seg_end)
        item_rgb = label_color

        row = [chrom, chrom_start, chrom_end, name, BED_SCORE, BED_STRAND,
               chrom_start, chrom_end, item_rgb][:num_cols]

        print >>outfile, "\t".join(row)

    # assert that the whole region is mapped
    # seg_end here means the last seg_end in the loop
    assert seg_end == region_end


def save_bed(outfilename, *args, **kwargs):
    with open(outfilename, "w") as outfile:
        write_bed(outfile, *args, **kwargs)


def read_posterior_save_bed(coord, resolution, do_reverse,
                            outfilename_tmpl, num_labels, infile,
                            num_sublabels, output_label):
    if do_reverse:
        raise NotImplementedError
    num_sublabels = int(num_sublabels)
    (chrom, start, end) = coord
    num_frames = ceildiv(end - start, resolution)
    probs = read_posterior(infile, num_frames, num_labels,
                           num_sublabels, output_label)
    probs_rounded = empty(probs.shape, float)  # casted to int after rounding

    # Write posterior code file
    posterior_code = argmax(probs, axis=1)
    if output_label != "seg":
        posterior_code = divide_posterior_array(posterior_code, num_frames,
                                                num_sublabels)
    start_pos, labels = find_segment_starts(posterior_code, output_label)
    bed_filename = outfilename_tmpl % "_code"
    save_bed(bed_filename, start_pos, labels, coord, resolution,
             int(num_labels))
    if output_label == "subseg":
        label_print_range = xrange(num_labels * num_sublabels)
        label_names = label_print_range
    elif output_label == "full":
        label_print_range = xrange(num_labels * num_sublabels)
        label_names = ("%d.%d" % divmod(label, num_sublabels)
                       for label in label_print_range)
    else:
        label_print_range = xrange(num_labels)
        label_names = label_print_range

    # Write label-wise posterior bedgraph files
    outfilenames = []
    for label_index in label_names:
        outfilenames.append(outfilename_tmpl % label_index)

    # scale, round, and cast to int
    (probs * POSTERIOR_SCALE_FACTOR).round(out=probs_rounded)
    probs_rounded = probs_rounded.astype(int)

    # print array columns as text to each outfile
    zipper = zip(outfilenames, probs_rounded.T, label_print_range)
    for outfilename, probs_rounded_label, label_index in zipper:
        # run-length encoding on the probs_rounded_label

        with open(outfilename, "w") as outfile:
            pos, = where(diff(probs_rounded_label) != 0)
            pos = r_[start, pos[:] + start + 1, end]

            for bed_start, bed_end in zip(pos[:-1], pos[1:]):
                chrom_start = str(bed_start)
                chrom_end = str(bed_end)
                value = str(probs_rounded_label[bed_start - start])

                row = [chrom, chrom_start, chrom_end, value]
                print >>outfile, "\t".join(row)


def load_posterior_save_bed(coord, resolution, do_reverse,
                            outfilename, num_labels, infilename,
                            num_sublabels=1, output_label="seg"):
    with open(infilename) as infile:
        read_posterior_save_bed(coord, resolution, do_reverse, outfilename,
                                int(num_labels), infile, num_sublabels,
                                output_label)


def parse_viterbi_save_bed(coord, resolution, do_reverse,
                           viterbi_lines, bed_filename, num_labels,
                           output_label):
    data = parse_viterbi(viterbi_lines, do_reverse, output_label)

    start_pos, labels = find_segment_starts(data, output_label)

    save_bed(bed_filename, start_pos, labels, coord, resolution,
             int(num_labels))


def load_viterbi_save_bed(coord, resolution, do_reverse, outfilename,
                          num_labels, infilename):
    with open(infilename) as infile:
        lines = infile.readlines()

    return parse_viterbi_save_bed(coord, resolution, do_reverse,
                                  lines, outfilename, num_labels)


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
        pass  # not going to add this filename to the command line
    temp_filepaths.append(filelistpath)

    return fd


def print_to_fd(fd, line):
    with fdopen(fd, "w") as outfile:
        print >>outfile, line


def run_posterior_save_bed(coord, resolution, do_reverse, outfilename,
                           num_labels, num_sublabels, output_label,
                           genomedata_names, float_filename, int_filename,
                           distribution, track_indexes_text, *args):
    # XXX: this whole function is duplicative of run_viterbi_save_bed
    # and needs to be reduced convert from tuple
    args = list(args)

    # a 2,000,000-frame output file is only 84 MiB so it is okay to
    # read the whole thing into memory
    (chrom, start, end) = coord
    track_indexes = make_track_indexes(track_indexes_text)

    float_filepath = TEMP_DIRPATH / float_filename
    int_filepath = TEMP_DIRPATH / int_filename
    temp_filepaths = [float_filepath, int_filepath]

    # XXX: should do something to ensure of1 matches with int, of2 with float
    float_filelistfd = replace_args_filelistname(args, temp_filepaths,
                                                 EXT_FLOAT)
    int_filelistfd = replace_args_filelistname(args, temp_filepaths, EXT_INT)

    with Genome(genomedata_names) as genome:
        continuous_cells = genome[chrom][start:end, track_indexes]

    try:
        print_to_fd(float_filelistfd, float_filename)
        print_to_fd(int_filelistfd, int_filename)

        _save_window(float_filename, int_filename, continuous_cells,
                     resolution, distribution)

        # XXXopt: does this actually free the memory? or do we need to
        # use a subprocess to do the loading?

        # remove from memory
        del continuous_cells

        output = POSTERIOR_PROG.getoutput(*args)
    finally:
        for filepath in temp_filepaths:
            # don't raise a nested exception if the file was never created
            try:
                filepath.remove()
            except OSError, err:
                if err.errno == ENOENT:
                    pass

    lines = output.splitlines()
    return read_posterior_save_bed(coord, resolution, do_reverse, outfilename,
                                   int(num_labels), lines,
                                   num_sublabels, output_label)


def run_viterbi_save_bed(coord, resolution, do_reverse, outfilename,
                         num_labels, num_sublabels, output_label,
                         genomedata_names, float_filename, int_filename,
                         distribution, track_indexes_text, *args):
    # convert from tuple
    args = list(args)
    # a 2,000,000-frame output file is only 84 MiB so it is okay to
    # read the whole thing into memory

    (chrom, start, end) = coord

    float_filepath = TEMP_DIRPATH / float_filename
    int_filepath = TEMP_DIRPATH / int_filename

    temp_filepaths = [float_filepath, int_filepath]

    # XXX: should do something to ensure of1 matches with int, of2 with float
    int_filelistfd = replace_args_filelistname(args, temp_filepaths, EXT_INT)
    float_filelistfd = replace_args_filelistname(args, temp_filepaths,
                                                 EXT_FLOAT)

    # Create a list of a list of tracks ordered based on genomedata archive
    # specified order (delimited with a ';')
    track_indexes = [make_track_indexes(genomedata_track_indexes_text)
                     for genomedata_track_indexes_text
                     in track_indexes_text.split(';')]
    genomedata_names = genomedata_names.split(",")

    continuous_cells = make_continuous_cells(track_indexes, genomedata_names,
                                             chrom, start, end)

    try:
        print_to_fd(float_filelistfd, float_filename)
        print_to_fd(int_filelistfd, int_filename)

        _save_window(float_filename, int_filename, continuous_cells,
                     resolution, distribution)

        # XXXopt: does this work? or do we need to use a subprocess to
        # do the loading?
        # remove from memory
        del continuous_cells

        output = VITERBI_PROG.getoutput(*args)
    finally:
        for filepath in temp_filepaths:
            # don't raise a nested exception if the file was never created
            try:
                filepath.remove()
            except OSError, err:
                if err.errno == ENOENT:
                    pass

    lines = output.splitlines()

    return parse_viterbi_save_bed(coord, resolution, do_reverse,
                                  lines, outfilename, num_labels, output_label)


def run_train(coord, resolution, do_reverse, outfilename,
              genomedata_names, float_filename, int_filename, distribution,
              track_indexes,
              is_semisupervised, supervision_coords, supervision_labels,
              *args):

    # Create and save the train window
    genomedata_names = genomedata_names.split(",")
    track_indexes = map(int, track_indexes.split(","))

    (chrom, start, end) = coord

    continuous_cells = make_continuous_cells(track_indexes, genomedata_names,
                                             chrom, start, end)

    # XXX: Currently disabled until dinucleotide is enabled
    # Only set these when dinucleotide is set
    # Get the first genome from this world to use for generating
    # sequence cells
    # with Genome(genomedata_names[0]) as genome:
    #     chromosome = genome[chrom]
    #     seq_cells = self.make_seq_cells(chromosome, start, end)

    # If this training regions is supervised
    if is_semisupervised == "True":
        # Create the supervision cells for this region
        # Convert supervision parameters back from text
        supervision_coords = literal_eval(supervision_coords)
        supervision_labels = literal_eval(supervision_labels)

        supervision_cells = make_supervision_cells(supervision_coords,
                                                   supervision_labels,
                                                   start, end)
    else:
        # Otherwise ignore supervision
        supervision_cells = None

    # Create observation file temporary paths
    float_filepath = TEMP_DIRPATH / float_filename
    int_filepath = TEMP_DIRPATH / int_filename

    temp_filepaths = [float_filepath, int_filepath]

    args = list(args)  # convert from tuple

    # Replace GMTK observation arguments with new temporary file paths
    int_filelistfd = replace_args_filelistname(args, temp_filepaths, EXT_INT)
    float_filelistfd = replace_args_filelistname(args, temp_filepaths,
                                                 EXT_FLOAT)
    # Set the range of observations to train over to only the first file in the
    # temporarily generated list
    args[args.index(GMTK_TRRNG_OPTION_STRING) + 1] = "0"

    try:
        print_to_fd(float_filelistfd, float_filename)
        print_to_fd(int_filelistfd, int_filename)

        # XXX: Add seq_data for dinucleotide (not impl)
        # XXX: Add exception handling on IOError?
        # Would have to parse error message?
        # e.g. Disk full vs No such file or directory
        _save_window(float_filename, int_filename, continuous_cells,
                     resolution, distribution,
                     supervision_data=supervision_cells)

        TRAIN_PROG.getoutput(*args)
    # After the training round is finished
    finally:
        # Remove all temporarily created files
        for filepath in temp_filepaths:
            # don't raise a nested exception if the file was never created
            try:
                filepath.remove()
            except OSError, err:
                if err.errno == ENOENT:
                    pass


def run_bundle_train(coord, resolution, do_reverse, outfilename, *args):
    # Create a list of temporary filenames
    temp_filepaths = []
    args = list(args)

    # Replace the GMTK observation list argument file with a temporarily
    # created one and append the filename to the list of temporary files
    int_filelistfd = replace_args_filelistname(args, temp_filepaths, EXT_INT)
    float_filelistfd = replace_args_filelistname(args, temp_filepaths,
                                                 EXT_FLOAT)

    # Try to write out a dummy observation filename to each observation file
    # list
    try:
        print_to_fd(float_filelistfd, DUMMY_OBSERVATION_FILENAME)
        print_to_fd(int_filelistfd, DUMMY_OBSERVATION_FILENAME)

        # Run EM bundling
        TRAIN_PROG.getoutput(*args)
    # After the EM bundling is finished
    finally:
        # Delete list of temporary files
        for filepath in temp_filepaths:
            # Don't raise a nested exception if the file was never created
            try:
                filepath.remove()
            except OSError, err:
                if err.errno == ENOENT:
                    pass


TASKS = {("run", "viterbi"): run_viterbi_save_bed,
         ("load", "viterbi"): load_viterbi_save_bed,
         ("run", "posterior"): run_posterior_save_bed,
         ("load", "posterior"): load_posterior_save_bed,
         ("run", "train"): run_train,
         ("run", "bundle-train"): run_bundle_train,
         }


def task(verb, kind, outfilename, chrom, start, end, resolution,
         reverse, *args):
    start = int(start)
    end = int(end)
    resolution = int(resolution)
    reverse = int(reverse)

    TASKS[verb, kind]((chrom, start, end), resolution, reverse,
                      outfilename, *args)


def main(args=sys.argv[1:]):
    if len(args) < 7:
        print >>sys.stderr, USAGE
        sys.exit(2)

    # Try running the task
    try:
        return task(*args)
    # If the there is an explicit out of memory exception
    except MemoryError:
        # return EX_TEMPFAIL error code
        return EX_TEMPFAIL
    # If the viterbi prog returns a non zero exit status
    except optbuild.ReturncodeError as return_code_exception:
        # return the non zero exit status
        return return_code_exception.returncode


if __name__ == "__main__":
    sys.exit(main())
