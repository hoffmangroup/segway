#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, with_statement

"""
task: wraps a GMTK subtask to reduce size of output
"""

# Copyright 2009-2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from ast import literal_eval
from contextlib import contextmanager
from errno import ENOENT
import gc
from os import extsep, fdopen, EX_TEMPFAIL, remove
import re
import sys
from tempfile import mkstemp

from numpy import argmax, array, empty, where, diff, r_, zeros
import optbuild
from six.moves import map, range, zip

from .observations import (make_continuous_cells, make_supervision_cells,
                           make_virtual_evidence_cells, _save_window)
from ._util import (BED_SCORE, BED_STRAND, ceildiv, DTYPE_IDENTIFY, EXT_FLOAT,
                    EXT_INT, EXT_LIST, EXT_VIRTUAL_EVIDENCE, extract_superlabel,
                    fill_array, find_segment_starts, get_label_color, TRAIN_PROG,
                    POSTERIOR_PROG, POSTERIOR_SCALE_FACTOR, read_posterior,
                    SEGWAY_ENCODING, VALIDATE_PROG, VITERBI_PROG,
                    VIRTUAL_EVIDENCE_LIST_FILENAME_PLACEHOLDER)

MSG_SUCCESS = "____ PROGRAM ENDED SUCCESSFULLY WITH STATUS 0 AT"

SCORE_MIN = 100
SCORE_MAX = 1000

SEG_INVALID = -1

EXT_OPTIONS = {}
EXT_OPTIONS[EXT_FLOAT] = "-of1"  # duplicative of run.py
EXT_OPTIONS[EXT_INT] = "-of2"

USAGE = "args: VERB KIND OUTFILE CHROM START END RESOLUTION REVERSE [ARGS...]"

# Dummy observation filename required for gmtk EM bundling. Has no effect but a
# name is necessary.
PLACEHOLDER_OBSERVATION_FILENAME = "/dev/zero"

GMTK_TRRNG_OPTION_STRING = "-trrng"  # Range to train over segment file


@contextmanager
def mkstemp_observation(chromosome_name, start, end, suffix):
    """A context manager that provides a tuple of a file object and full
    filepath of an a observation file with the given suffix.
    Does not delete the created temporary file after exiting."""

    # Set a common prefix for the observation files
    prefix = "{}.{}.{}.".format(chromosome_name, start, end)
    fd, filename = mkstemp(suffix=extsep + suffix, prefix=prefix)

    temp_observation_file = fdopen(fd, "w")

    yield temp_observation_file, filename

    temp_observation_file.close()


def save_temp_observations(chromosome_name, start, end, continuous_cells,
                           resolution, distribution, supervision_data,
                           virtual_evidence_data, num_labels):
    """Returns a tuple (float_obs, int_obs) of temporary filepaths for the
    int/float observation filenames unique to this process"""

    # Create secure temporary observation files
    with mkstemp_observation(chromosome_name, start, end, EXT_FLOAT) as \
            (float_observations_file, float_observations_filename), \
            mkstemp_observation(chromosome_name, start, end, EXT_INT) as \
            (int_observations_file, int_observations_filename), \
            mkstemp_observation(chromosome_name, start, end, EXT_VIRTUAL_EVIDENCE) as \
            (virtual_evidence_file, virtual_evidence_filename):

            # numpy's tofile (which is used) can take an open python file
            # object
            # XXX: Currently seq_data is disabled until dinucleotide is enabled
            _save_window(float_observations_file, int_observations_file,
                         continuous_cells, resolution, distribution,
                         seq_data=None, supervision_data=supervision_data,
                         virtual_evidence_data=virtual_evidence_data,
                         virtual_evidence_filename_or_file=virtual_evidence_file,
                         num_labels=num_labels)

    return float_observations_filename, int_observations_filename, \
           virtual_evidence_filename


def save_temp_observation_filelists(float_observations_filename,
                                    int_observations_filename,
                                    virtual_evidence_observations_filename):
    """Create an observation file list containing the respective observation
    file name.

    Returns a tuple (float_obs_list, int_obs_list) of temporary filepaths
    for the int/float observation lists (files) unique to this process
    """

    # Create secure temporary observation files
    float_observation_list_fd, float_observation_list_filename = \
        mkstemp(prefix=EXT_FLOAT + extsep, suffix=extsep + EXT_LIST)
    int_observation_list_fd, int_observation_list_filename = \
        mkstemp(prefix=EXT_INT + extsep, suffix=extsep + EXT_LIST)
    virtual_evidence_observation_list_fd,\
        virtual_evidence_observation_list_filename = \
        mkstemp(prefix=EXT_VIRTUAL_EVIDENCE + extsep, suffix=extsep + EXT_LIST)
    # Write out the observation filename to their respective observation list
    # For gmtk observation list files, there may be more than one
    # observation file. In this case we only ever insert one
    # print_to_fd uses a context manager which implicity closes the
    # os-level file descriptor
    print_to_fd(float_observation_list_fd, float_observations_filename)
    print_to_fd(int_observation_list_fd, int_observations_filename)
    print_to_fd(virtual_evidence_observation_list_fd,
                virtual_evidence_observations_filename)

    return float_observation_list_filename, int_observation_list_filename, \
           virtual_evidence_observation_list_filename


def replace_subsequent_value(input_list, query, new):
    """Attempts to modify the given input list with no exception so that the
    value following the query is modified to the new value """

    try:
        new_index = input_list.index(query) + 1
        input_list[new_index] = new
    # If the query value is not found
    except ValueError:
        # Do nothing
        pass
    # If the new index is out of range
    except IndexError:
        # Do nothing
        pass

def prepare_virtual_evidence(virtual_evidence, start, end, num_labels,
                             virtual_evidence_coords, virtual_evidence_priors):
    if virtual_evidence == "False":
        return None
    virtual_evidence_coords = literal_eval(virtual_evidence_coords)
    virtual_evidence_priors = literal_eval(virtual_evidence_priors)

    virtual_evidence_cells = make_virtual_evidence_cells(
               virtual_evidence_coords,
               virtual_evidence_priors,
               start, end, num_labels)

    return virtual_evidence_cells

def prepare_gmtk_observations(gmtk_args, chromosome_name, start, end,
                              continuous_cells, resolution, distribution,
                              supervision_data=None, virtual_evidence_data=None,
                              num_labels=None):
    """Returns a list of filepaths to observation files created for gmtk
    and modifies the necessary arguments (args) for running gmtk"""

    try:
        # Create the gmtk observation files
        float_observations_filename, int_observations_filename, \
            virtual_evidence_filename = \
            save_temp_observations(chromosome_name, start, end,
                                   continuous_cells, resolution, distribution,
                                   supervision_data, virtual_evidence_data,
                                   num_labels)

        # Create the gmtk observation file lists
        float_observation_list_filename, int_observation_list_filename, \
            virtual_evidence_list_filename = \
            save_temp_observation_filelists(float_observations_filename,
                                            int_observations_filename,
                                            virtual_evidence_filename)
    # If any exception occurred
    except:  # NOQA
        # Attempt to remove any created files
        force_remove_all_files([float_observations_filename,
                                int_observations_filename,
                                virtual_evidence_filename,
                                float_observation_list_filename,
                                int_observation_list_filename,
                                virtual_evidence_list_filename])
        # Reraise the exception
        raise

    # Modify the given gmtk arguments to use the temporary observation lists
    replace_subsequent_value(gmtk_args, EXT_OPTIONS[EXT_FLOAT],
                             float_observation_list_filename)
    replace_subsequent_value(gmtk_args, EXT_OPTIONS[EXT_INT],
                             int_observation_list_filename)

    # cppCommandOptions is stored as a string with format CPP_DIRECTIVE_FMT
    cpp_command_options_index = gmtk_args.index("-cppCommandOptions") + 1
    cpp_command_str = gmtk_args[cpp_command_options_index]
    # if the placeholder is present, it is replaced. otherwise, the cpp options
    # are unchanged
    gmtk_args[cpp_command_options_index] = cpp_command_str.replace(
                                    VIRTUAL_EVIDENCE_LIST_FILENAME_PLACEHOLDER,
                                    virtual_evidence_list_filename, 1)

    # Modify the given gmtk arguments so only the first (and only) file in the
    # observation lists are used
    replace_subsequent_value(gmtk_args, GMTK_TRRNG_OPTION_STRING, "0")

    # Return the list of filenames created
    return [float_observations_filename, int_observations_filename,
            virtual_evidence_filename,
            float_observation_list_filename, int_observation_list_filename,
            virtual_evidence_list_filename]


@contextmanager
def files_to_remove(filenames):
    """Creates a context manager where upon exit, ensures files are removed"""
    yield

    force_remove_all_files(filenames)


def force_remove_all_files(filenames):
    exception_list = []
    # For each file name
    for filename in filenames:
        # Attempt to remove the file
        try:
            force_remove_file(filename)
        # Catch any exception
        except:  # NOQA
            # Store the exception
            # The 2nd element from exc_info is the exception object instance
            exception_list.append(sys.exc_info()[1])

    handle_multiple_exceptions(exception_list)


def force_remove_file(filename):
    """Attempts to remove the given filename. Catches and ignores an exception
    if the file does not exist and raises all other exceptions"""
    try:
        remove(filename)
    # Ignore exceptions where the file does not exist
    except OSError as err:
        # If a different exception was found
        if err.errno != ENOENT:
            # Reraise
            raise


def handle_multiple_exceptions(exception_list):
    """Takes an list of exception objects and prints out all information to
    stderr and raises the first exception (if any exceptions exist)."""
    # If any exceptions exist
    if exception_list:
        # Print out all exceptions and raise the first
        first_exception = exception_list.pop(0)
        for exception in exception_list:
            print(exception, file=sys.stderr)
        raise first_exception


def make_track_indexes(text):
    return array(text.split(","), int)


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
    for frame_index in range(num_frames):
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
    assert next(lines).startswith("Segment ")

    # ========
    assert next(lines).startswith("========")

    # Segment 0, number of frames = 1001, viteri-score = -6998.363710
    line = next(lines)
    assert line.startswith("Segment ")

    num_frames_text = line.split(", ")[1].partition(" = ")
    assert num_frames_text[0] == "number of frames"
    assert num_frames_text[1] == " = "

    num_frames = int(num_frames_text[2])

    # Printing random variables from (P,C,E)=(1,999,0) partitions
    line = next(lines)
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

        print(*row, sep="\t", file=outfile)

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
        label_print_range = range(num_labels * num_sublabels)
        label_names = label_print_range
    elif output_label == "full":
        label_print_range = range(num_labels * num_sublabels)
        label_names = ("%d.%d" % divmod(label, num_sublabels)
                       for label in label_print_range)
    else:
        label_print_range = range(num_labels)
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
            # Create a list of indicies of unique values in the probability
            # BED labels
            unique_prob_value_indices, = where(diff(probs_rounded_label) != 0)
            # The first index is always unique
            unique_prob_value_indices = r_[0, unique_prob_value_indices + 1]
            region_coords = r_[(unique_prob_value_indices * resolution) +
                               start,
                               end]

            bed_zipper = zip(region_coords[:-1], region_coords[1:],
                             unique_prob_value_indices)

            for bed_start, bed_end, prob_index in bed_zipper:
                chrom_start = str(bed_start)
                chrom_end = str(bed_end)
                value = str(probs_rounded_label[prob_index])

                row = [chrom, chrom_start, chrom_end, value]
                print(*row, sep="\t", file=outfile)


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


def print_to_fd(fd, line):
    with fdopen(fd, "w") as outfile:
        print(line, file=outfile)


def run_posterior_save_bed(coord, resolution, do_reverse, outfilename,
                           num_labels, num_sublabels, output_label,
                           genomedata_names, distribution, track_indexes_text,
                           virtual_evidence, virtual_evidence_coords,
                           virtual_evidence_priors, *args):
    # XXX: this whole function is duplicative of run_viterbi_save_bed
    # and needs to be reduced convert from tuple
    args = list(args)

    # a 2,000,000-frame output file is only 84 MiB so it is okay to
    # read the whole thing into memory
    (chrom, start, end) = coord
    # Create and save the window
    genomedata_names = genomedata_names.split(",")
    track_indexes = make_track_indexes(track_indexes_text)

    continuous_cells = make_continuous_cells(track_indexes, genomedata_names,
                                             chrom, start, end)

    num_labels = literal_eval(num_labels)
    virtual_evidence_cells = prepare_virtual_evidence(virtual_evidence,
                                                      start, end, num_labels,
                                                      virtual_evidence_coords,
                                                      virtual_evidence_priors)

    temp_filenames = prepare_gmtk_observations(args, chrom, start, end,
                                               continuous_cells,
                                               resolution, distribution,
                                               None, virtual_evidence_cells,
                                               num_labels)
    # remove from memory
    del continuous_cells
    gc.collect()

    with files_to_remove(temp_filenames):
        output = POSTERIOR_PROG.getoutput(*args)

    lines = output.splitlines()
    return read_posterior_save_bed(coord, resolution, do_reverse, outfilename,
                                   int(num_labels), lines,
                                   num_sublabels, output_label)


def run_viterbi_save_bed(coord, resolution, do_reverse, outfilename,
                         num_labels, num_sublabels, output_label,
                         genomedata_names, distribution, track_indexes_text,
                         virtual_evidence, virtual_evidence_coords,
                         virtual_evidence_priors, *args):
    # convert from tuple
    args = list(args)
    # a 2,000,000-frame output file is only 84 MiB so it is okay to
    # read the whole thing into memory

    (chrom, start, end) = coord

    # Create and save the window
    genomedata_names = genomedata_names.split(",")
    track_indexes = make_track_indexes(track_indexes_text)

    continuous_cells = make_continuous_cells(track_indexes, genomedata_names,
                                             chrom, start, end)

    num_labels = literal_eval(num_labels)
    virtual_evidence_cells = prepare_virtual_evidence(virtual_evidence,
                                                      start, end, num_labels,
                                                      virtual_evidence_coords,
                                                      virtual_evidence_priors)

    temp_filenames = prepare_gmtk_observations(args, chrom, start, end,
                                               continuous_cells,
                                               resolution, distribution,
                                               None, virtual_evidence_cells,
                                               num_labels)
    # remove from memory
    del continuous_cells
    gc.collect()

    with files_to_remove(temp_filenames):
        output = VITERBI_PROG.getoutput(*args)

    lines = [line.decode(SEGWAY_ENCODING) for line in output.splitlines()]

    return parse_viterbi_save_bed(coord, resolution, do_reverse,
                                  lines, outfilename, num_labels, output_label)


def run_train(coord, resolution, do_reverse, outfilename,
              genomedata_names, distribution,
              track_indexes,
              is_semisupervised, supervision_coords, supervision_labels,
              virtual_evidence, virtual_evidence_coords,
              virtual_evidence_priors, num_labels, *args):

    # Create and save the train window
    genomedata_names = genomedata_names.split(",")
    track_indexes = make_track_indexes(track_indexes)

    (chrom, start, end) = coord
    gmtk_args = list(args)

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

    num_labels = literal_eval(num_labels)
    virtual_evidence_cells = prepare_virtual_evidence(virtual_evidence,
                                                      start, end, num_labels,
                                                      virtual_evidence_coords,
                                                      virtual_evidence_priors)

    temp_filenames = prepare_gmtk_observations(gmtk_args, chrom, start,
                                               end, continuous_cells,
                                               resolution, distribution,
                                               supervision_cells,
                                               virtual_evidence_cells,
                                               num_labels)
    del continuous_cells
    gc.collect()

    with files_to_remove(temp_filenames):
        TRAIN_PROG.run(*gmtk_args)


def run_bundle_train(coord, resolution, do_reverse, outfilename, *args):
    args = list(args)

    # Create placeholder observation lists
    placeholder_float_list, placeholder_int_list, \
        placeholder_virtual_evidence_list = \
        save_temp_observation_filelists(PLACEHOLDER_OBSERVATION_FILENAME,
                                        PLACEHOLDER_OBSERVATION_FILENAME,
                                        PLACEHOLDER_OBSERVATION_FILENAME)
    # Modify the given gmtk arguments to use the temporary placeholder
    # observation lists
    replace_subsequent_value(args, EXT_OPTIONS[EXT_FLOAT],
                             placeholder_float_list)
    replace_subsequent_value(args, EXT_OPTIONS[EXT_INT], placeholder_int_list)

    # cppCommandOptions is stored as a string with format CPP_DIRECTIVE_FMT
    cpp_command_options_index = args.index("-cppCommandOptions") + 1
    cpp_command_str = args[cpp_command_options_index]

    # if the placeholder is present, it is replaced. otherwise, the cpp options
    # are unchanged
    args[cpp_command_options_index] = cpp_command_str.replace(
                                    VIRTUAL_EVIDENCE_LIST_FILENAME_PLACEHOLDER,
                                    placeholder_virtual_evidence_list, 1)
    # Run EM bundling
    with files_to_remove([placeholder_float_list, placeholder_int_list]):
        TRAIN_PROG.getoutput(*args)


def save_gmtk_observation_files(coord, resolution, do_reverse, outfile_name,
                                genomedata_names, float_filename,
                                int_filename, distribution,
                                track_indexes, *args):

    (chrom, start, end) = coord

    genomedata_names = genomedata_names.split(",")
    track_indexes = map(int, track_indexes.split(","))

    continuous_cells = make_continuous_cells(track_indexes, genomedata_names,
                                             chrom, start, end)

    _save_window(float_filename, int_filename, continuous_cells,
                 resolution, distribution)


def run_validate(coord, resolution, do_reverse, outfilename, *args):
    if do_reverse:
        raise NotImplementedError("Running Segway with both validation and "
                                  "reverse world options simultaneously is "
                                  "currently not supported")

    validation_output = VALIDATE_PROG.getoutput(*args).decode(SEGWAY_ENCODING)
    with open(outfilename, "w") as outfile:
        outfile.write(validation_output)


TASKS = {("run", "viterbi"): run_viterbi_save_bed,
         ("load", "viterbi"): load_viterbi_save_bed,
         ("run", "posterior"): run_posterior_save_bed,
         ("load", "posterior"): load_posterior_save_bed,
         ("run", "train"): run_train,
         ("run", "bundle-train"): run_bundle_train,
         ("run", "validate"): run_validate,
         ("save", "gmtk-observation-files"): save_gmtk_observation_files,
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
        print(USAGE, file=sys.stderr)
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
