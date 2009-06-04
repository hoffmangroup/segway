#!/usr/bin/env python
from __future__ import division, with_statement

"""
task: wraps a GMTK subtask to reduce size of output
"""

__version__ = "$Revision$"

# Copyright 2009 Michael M. Hoffman <mmh1@washington.edu>

import re
import sys

from numpy import zeros

from ._util import (DTYPE_IDENTIFY, find_segment_starts, get_label_color,
                    VITERBI_PROG)

MSG_SUCCESS = "____ PROGRAM ENDED SUCCESSFULLY WITH STATUS 0 AT"

BED_SCORE = "1000"
BED_STRAND = "."

SCORE_MIN = 100
SCORE_MAX = 1000

re_seg = re.compile(r"^seg\((\d+)\)=(\d+)$")
def parse_viterbi(lines):
    lines = iter(lines)

    assert lines.next().startswith("Segment ")
    assert lines.next().startswith("========")

    line = lines.next()
    assert line.startswith("Segment ")

    num_frames_text = line.split(", ")[1].partition(" = ")
    assert num_frames_text[0] == "number of frames"
    assert num_frames_text[1] == " = "

    num_frames = int(num_frames_text[2])

    line = lines.next()
    assert line.startswith("Printing random variables from (P,C,E)")

    res = zeros(num_frames, DTYPE_IDENTIFY)

    for line in lines:
        if line.startswith(MSG_SUCCESS):
            return res

        assert line.rstrip().endswith(" partition")

        line = lines.next()
        for pair in line.split(","):
            match = re_seg.match(pair)
            if not match:
                continue

            index = int(match.group(1))
            val = int(match.group(2))

            res[index] = val

    # shouldn't get to this point
    raise ValueError("%s did not complete successfully" % VITERBI_PROG.prog)

# num_cols is for use by genomedata_count_condition
# XXX: should move this function somewhere else
def write_bed(outfile, start_pos, labels, coord, num_labels, num_cols=None):
    (chrom, region_start, region_end) = coord

    start_pos += region_start

    score_step = (SCORE_MAX - SCORE_MIN) / (num_labels - 1)

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

def parse_viterbi_save_bed(coord, viterbi_lines, bed_filename, num_labels):
    data = parse_viterbi(viterbi_lines)

    start_pos, labels = find_segment_starts(data)

    save_bed(bed_filename, start_pos, labels, coord, int(num_labels))

def load_viterbi_save_bed(coord, outfilename, num_labels, infilename):
    with open(infilename) as infile:
        lines = infile.readlines()

    return parse_viterbi_save_bed(coord, lines, outfilename, num_labels)

def run_viterbi_save_bed(coord, outfilename, num_labels, *args):
    # a 2,000,000-frame output file is only 84 MiB so it is okay to
    # read the whole thing into memory

    output = VITERBI_PROG.getoutput(*args)
    lines = output.splitlines()

    return parse_viterbi_save_bed(coord, lines, outfilename, num_labels)

TASKS = {("run", "viterbi"): run_viterbi_save_bed,
         ("load", "viterbi"): load_viterbi_save_bed}

def task(verb, kind, outfilename, chrom, start, end, *args):
    start = int(start)
    end = int(end)

    TASKS[verb, kind]((chrom, start, end), outfilename, *args)

def main(args=sys.argv[1:]):
    if len(args) < 6:
        print >>sys.stderr, \
            "args: VERB KIND OUTFILE CHROM START END [ARGS...]"
        sys.exit(2)

    return task(*args)

if __name__ == "__main__":
    sys.exit(main())
