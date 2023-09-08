#!/usr/bin/env python
from __future__ import division, print_function

"""output.py: output savers: IdentifySaver, PosteriorSaver
"""

__version__ = "$Revision$"

## Copyright 2012, 2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from six.moves import range

from .bed import parse_bed4
from .layer import layer, make_layer_filename
from ._util import (Copier, extract_superlabel, get_label_color, 
                    maybe_gzip_open, BED_SCORE, BED_STRAND, 
                    INDEX_BED_CHROMSTART, INDEX_BED_NAME, INDEX_BED_THICKSTART,
                    NUM_COLORS)


def make_bed_attr(key, value):
    if " " in value:
        value = '"%s"' % value

    return "%s=%s" % (key, value)


def make_bed_attrs(mapping):
    res = " ".join(make_bed_attr(key, mapping[key])
                   for key in sorted(mapping))

    return "track %s" % res


def make_label_colors(window_filenames):
    """
    Create a color assignment for each superlabel in the window files.
    Numerical labels are treated as indices into the color palette list.
    String labels are sorted in alphabetical order and the first colors in the 
    color palette are used.

    Example color assignment for labels "a", "b", "c": 
    {"a": "27,158,119", "b": "217,95,2", "c": "117,112,179"}
    """

    # 
    labels = set()
    for window_filename in window_filenames:
        with maybe_gzip_open(window_filename) as window_file:
            for line in window_file:
                label = extract_superlabel(line.split()[INDEX_BED_NAME])
                labels.add(label)

    # If labels are numbers, use as indices into color list
    if all(label.isdecimal() for label in labels): # Index into color list
        return {label: get_label_color(int(label))
                for label in labels}
    
    # Otherwise, assign colors by alphabetical order
    return {label: get_label_color(index) 
            for index, label in enumerate(sorted(list(labels)))}


def merge_windows_to_bed(window_filenames, header, bedfilename, trackfilename,
                         bed9_output=True):
    """
    Given a list of filepaths, each of which points to a segmentation BED4 file,
    concatenate the files and merge entries if necessary. 
    Write two files: a BED4 file and a track file containing a track line and 
    BED data.
    """
    # If converting to 9-column output formats, build a color 
    # assignment using all labels
    if bed9_output:
        label_colors = make_label_colors(window_filenames)

    # values for comparison to combine adjoining segments
    last_line = ""
    last_start = None
    last_vals = (None, None, None) # (chrom, coord, seg)

    with maybe_gzip_open(bedfilename, "wt") as bedfile, maybe_gzip_open(trackfilename, "wt") as trackfile:
        # Write header to trackfile only
        print(header, file=trackfile)

        for window_filename in window_filenames:
            with maybe_gzip_open(window_filename) as window_file:
                lines = window_file.readlines()
                
                # If converting to 9 column format, add extra columns 
                # to BED4 data
                if bed9_output:
                    for index, line in enumerate(lines):
                        _, coords = parse_bed4(line)
                        (chrom, start, end, seg) = coords
                        bed9row = [chrom, start, end, seg, BED_SCORE, 
                                BED_STRAND, start, end, 
                                label_colors[extract_superlabel(seg)]]
                        lines[index] = '\t'.join(bed9row) + '\n'

                # Prepare the first line of the data
                first_line = lines[0]
                first_row, first_coords = parse_bed4(first_line)
                (chrom, start, end, seg) = first_coords

                # write the last line and the first line, after
                # potentially merging
                if last_vals == (chrom, start, seg):
                    # update start position
                    first_row[INDEX_BED_CHROMSTART] = last_start

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
                _, last_coords = parse_bed4(last_line)
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

    def merge_windows(self, world):
        # the final bed filename, not the individual viterbi_filenames
        bedfilename = self.make_filename(self.bed_filename, world)
        trackfilename = self.make_filename(self.track_filename, world)
        windows = self.windows

        world_viterbi_filenames = [viterbi_filename
                                   for window_index, viterbi_filename
                                   in enumerate(self.viterbi_filenames)
                                   if windows[window_index].world == world]
        header = self.make_track_header()
        merge_windows_to_bed(world_viterbi_filenames, header, 
                             bedfilename, trackfilename)

    def __call__(self, world):
        self.merge_windows(world)

        track_filename = self.make_filename(self.track_filename, world)
        layer(track_filename, make_layer_filename(track_filename),
              bigbed_outfilename=self.make_filename(self.bigbed_filename, world))


class PosteriorSaver(OutputSaver):
    copy_attrs = OutputSaver.copy_attrs + ["bedgraph_filename", 
                                           "posterior_bed_filename",
                                           "bed_filename",
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
        # Save posterior code bed and track files
        posterior_code_bedfilename = self.make_filename(self.bed_filename, world)
        posterior_code_trackfilename = self.make_filename(self.track_filename, world)
        posterior_code_filenames = [posterior_tmpl % "_code" for posterior_tmpl in self.posterior_filenames]
        trackheader = self.make_track_header()
        merge_windows_to_bed(posterior_code_filenames, trackheader, 
                             posterior_code_bedfilename,
                             posterior_code_trackfilename)

        # Save posterior bed and bedgraph files
        posterior_bed_tmpl = self.make_filename(self.posterior_bed_filename, world)
        posterior_bedgraph_tmpl = self.make_filename(self.bedgraph_filename, 
                                                     world)
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
            merge_windows_to_bed(posterior_filenames, trackheader, 
                                 posterior_bed_tmpl % num_seg,
                                 posterior_bedgraph_tmpl % num_seg,
                                 bed9_output = False)
