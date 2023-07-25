#!/usr/bin/env python
from __future__ import division, print_function

"""output.py: output savers: IdentifySaver, PosteriorSaver
"""

__version__ = "$Revision$"

## Copyright 2012, 2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from six.moves import range
import colorbrewer

from .bed import parse_bed4
from .layer import layer, make_layer_filename
from ._util import Copier, maybe_gzip_open, BED_SCORE, BED_STRAND

INDEX_BED_START = 1
INDEX_BED_THICKSTART = INDEX_BED_START + 5

NUM_COLORS = 8
SCHEME = colorbrewer.Dark2[NUM_COLORS]

def make_bed_attr(key, value):
    if " " in value:
        value = '"%s"' % value

    return "%s=%s" % (key, value)

def make_bed_attrs(mapping):
    res = " ".join(make_bed_attr(key, mapping[key])
                   for key in sorted(mapping))

    return "track %s" % res


# Takes a list of filepaths, each of which points to a segmentation bed file.
# Concatenates the files, merging entries if necessary.  Start file with
# header.  Used by IdentifySaver and PosteriorSaver.
def concatenate_window_segmentations(window_filenames, header, 
                                     bedfilename, trackfilename):
    # Iterate through window files to build a label list
    labels = set()
    for window_filename in window_filenames:
        with maybe_gzip_open(window_filename) as window_file:
            for line in window_file:
                labels.add(line.split()[3])
    # Build a color assignment from the label list
    label2color = {label: ','.join(map(str, SCHEME[(i + 1) % NUM_COLORS])) 
                   for i, label in enumerate(sorted(list(labels)))}

    # values for comparison to combine adjoining segments
    last_line = ""
    last_start = None
    last_vals = (None, None, None) # (chrom, coord, seg)

    with maybe_gzip_open(bedfilename, "wt") as bedfile:
        with maybe_gzip_open(trackfilename, "wt") as trackfile:
            # XXX: add in browser track line (see SVN revisions
            # previous to 195)
            print(header, file=trackfile)

            for window_filename in window_filenames:
                with maybe_gzip_open(window_filename) as window_file:
                    lines = window_file.readlines()
                    # Created BED9 lines from the BED4 data
                    for i, line in enumerate(lines):
                        _, coords = parse_bed4(line)
                        (chrom, start, end, seg) = coords
                        bed9row = [chrom, start, end, seg, BED_SCORE, 
                                   BED_STRAND, start, end, label2color[seg]]
                        lines[i] = '\t'.join(bed9row) + '\n'

                    # Prepare the first line of the BED9 data
                    first_line = lines[0]
                    first_row, first_coords = parse_bed4(first_line)
                    (chrom, start, end, seg) = first_coords

                    # write the last line and the first line, after
                    # potentially merging
                    if last_vals == (chrom, start, seg):
                        # update start position
                        first_row[INDEX_BED_START] = last_start
                        # update thickStart position
                        first_row[INDEX_BED_THICKSTART] = last_start

                        # add back trailing newline eliminated by line.split()
                        merged_line = "\t".join(first_row) + "\n"

                        # if there's just a single line in the BED file
                        if len(lines) == 1:
                            last_line = merged_line
                            last_vals = (chrom, end, seg)
                            # last_start is already set correctly
                            # postpone writing until after additional merges
                            continue
                        else:
                            # write the merged line
                            bedfile.write(merged_line)
                            trackfile.write(merged_line)
                    else:
                        if len(lines) == 1:
                            # write the last line of the last file.
                            # hold back the first line of this file,
                            # and treat it as the last line
                            bedfile.write(last_line)
                            trackfile.write(last_line)
                        else:
                            # write the last line of the last file, first
                            # line of this file
                            bedfile.writelines([last_line, first_line])
                            trackfile.writelines([last_line, first_line])

                    # write the bulk of the lines
                    bedfile.writelines(lines[1:-1])
                    trackfile.writelines(lines[1:-1])

                    # set last_line
                    last_line = lines[-1]
                    last_row, last_coords = parse_bed4(last_line)
                    (chrom, start, end, seg) = last_coords
                    last_vals = (chrom, end, seg)
                    last_start = start

            # write the very last line of all files
            bedfile.write(last_line)
            trackfile.write(last_line)

class OutputSaver(Copier):
    copy_attrs = ["tracks", "uuid", "num_worlds", "num_segs", "num_subsegs"]

    attrs = dict(visibility="dense",
                 viewLimits="0:1",
                 itemRgb="on",
                 autoScale="off")
    name_tmpl = "%s.%s"
    desc_tmpl = "%s %%d-label segmentation of %%s" % __package__

    def make_filename(self, fmt, world):
        """
        if there are multiple worlds, do another level of substitution
        """
        if not fmt or self.num_worlds == 1:
            return fmt

        return fmt % world

    def make_track_header(self):
        attrs = self.attrs.copy()
        attrs["name"] = self.name_tmpl % (__package__, self.uuid)

        tracknames = ", ".join(track.name_unquoted for track in self.tracks)
        description = self.desc_tmpl % (self.num_segs, tracknames)
        attrs["description"] = description

        return make_bed_attrs(attrs)

class IdentifySaver(OutputSaver):
    copy_attrs = OutputSaver.copy_attrs + ["bed_filename", "track_filename", "viterbi_filenames", "bigbed_filename", "windows"]

    def get_world_indexes(self, world):
        return [index
                for index, window in enumerate(self.windows)
                if world == window.world]

    def concatenate(self, world):
        # the final bed filename, not the individual viterbi_filenames
        bedfilename = self.make_filename(self.bed_filename, world)
        trackfilename = self.make_filename(self.track_filename, world)
        windows = self.windows

        world_viterbi_filenames = [viterbi_filename
                                   for window_index, viterbi_filename
                                   in enumerate(self.viterbi_filenames)
                                   if windows[window_index].world == world]
        header = self.make_track_header()
        concatenate_window_segmentations(world_viterbi_filenames, header, 
                                         bedfilename, trackfilename)

    def __call__(self, world):
        self.concatenate(world)

        track_filename = self.make_filename(self.track_filename, world)
        layer(track_filename, make_layer_filename(track_filename),
              bigbed_outfilename=self.make_filename(self.bigbed_filename, world))


class PosteriorSaver(OutputSaver):
    copy_attrs = OutputSaver.copy_attrs + ["bedgraph_filename", "bed_filename",
                                           "track_filename",
                                           "posterior_filenames",
                                           "output_label"]

    def make_bedgraph_header(self, num_seg):
        if self.output_label == "full":
            bedgraph_header_tmpl = "track type=bedGraph name=posterior.%s \
            description=\"Segway posterior probability of label %s\" \
            visibility=dense  viewLimits=0:100 maxHeightPixels=0:0:10 \
            autoScale=off color=200,100,0 altColor=0,100,200"
        else:
            bedgraph_header_tmpl = "track type=bedGraph name=posterior.%d \
            description=\"Segway posterior probability of label %d\" \
            visibility=dense  viewLimits=0:100 maxHeightPixels=0:0:10 \
            autoScale=off color=200,100,0 altColor=0,100,200"
        return bedgraph_header_tmpl % (num_seg, num_seg)

    def __call__(self, world):
        # Save posterior code bed file
        posterior_code_bedfilename = self.make_filename(self.bed_filename, world)
        posterior_code_trackfilename = self.make_filename(self.track_filename, world)
        posterior_code_filenames = [posterior_tmpl % "_code" for posterior_tmpl in self.posterior_filenames]
        trackheader = self.make_track_header()
        concatenate_window_segmentations(posterior_code_filenames, trackheader, 
                                         posterior_code_bedfilename, 
                                         posterior_code_trackfilename)

        # Save posterior bedgraph files
        posterior_bedgraph_tmpl = self.make_filename(self.bedgraph_filename, world)
        if self.output_label == "subseg":
            label_print_range = range(self.num_segs * self.num_subsegs)
        elif self.output_label == "full":
            label_print_range = ("%d.%d" % divmod(label, self.num_subsegs)
                                 for label in range(self.num_segs *
                                                     self.num_subsegs))
        else:
            label_print_range = range(self.num_segs)
        for num_seg in label_print_range:
            posterior_filenames = [posterior_tmpl % num_seg for posterior_tmpl in self.posterior_filenames]
            trackheader = self.make_bedgraph_header(num_seg)
            concatenate_window_segmentations(posterior_filenames, trackheader, 
                                             "/dev/null",
                                             posterior_bedgraph_tmpl % num_seg)
