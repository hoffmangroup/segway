#!/usr/bin/env python
from __future__ import division

"""output.py: output savers: IdentifySaver, PosteriorSaver
"""

__version__ = "$Revision$"

## Copyright 2012 Michael M. Hoffman <mmh1@uw.edu>

from .bed import parse_bed4
from .layer import layer, make_layer_filename
from ._util import Copier, maybe_gzip_open

INDEX_BED_START = 1

def make_bed_attr(key, value):
    if " " in value:
        value = '"%s"' % value

    return "%s=%s" % (key, value)

def make_bed_attrs(mapping):
    res = " ".join(make_bed_attr(key, value)
                   for key, value in mapping.iteritems())

    return "track %s" % res

class OutputSaver(Copier):
    def make_filename(self, fmt, world):
        """
        if there are multiple worlds, do another level of substitution
        """
        if not fmt or self.num_worlds == 1:
            return fmt

        return fmt % world

class IdentifySaver(OutputSaver):
    copy_attrs = ["bed_filename", "unquoted_tracknames", "uuid",
                  "viterbi_filenames", "bigbed_filename", "window_coords",
                  "num_worlds"]

    attrs = dict(visibility="dense",
                 viewLimits="0:1",
                 itemRgb="on",
                 autoScale="off")

    name_tmpl = "%s.%s"
    desc_tmpl = "%s segmentation of %%s" % __package__

    def make_header(self):
        attrs = self.attrs.copy()
        attrs["name"] = self.name_tmpl % (__package__, self.uuid)

        description = self.desc_tmpl % ", ".join(self.unquoted_tracknames)
        attrs["description"] = description

        return make_bed_attrs(attrs)

    def get_world_indexes(self, world):
        return [index
                for index, (window_world, chrom, start, end)
                in enumerate(self.window_coords)
                if world == window_world]

    def concatenate(self, world):
        # the final bed filename, not the individual viterbi_filenames
        outfilename = self.make_filename(self.bed_filename, world)
        window_coords = self.window_coords

        # values for comparison to combine adjoining segments
        last_line = ""
        last_start = None
        last_vals = (None, None, None) # (chrom, coord, seg)

        with maybe_gzip_open(outfilename, "w") as outfile:
            # XXX: add in browser track line (see SVN revisions
            # previous to 195)
            print >>outfile, self.make_header()

            for window_index, viterbi_filename in \
                    enumerate(self.viterbi_filenames):
                if window_coords[window_index][0] != world:
                    continue

                with open(viterbi_filename) as viterbi_file:
                    lines = viterbi_file.readlines()
                    first_line = lines[0]
                    first_row, first_coords = parse_bed4(first_line)
                    (chrom, start, end, seg) = first_coords

                    # write the last line and the first line, after
                    # potentially merging
                    if last_vals == (chrom, start, seg):
                        first_row[INDEX_BED_START] = last_start

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
                            outfile.write(merged_line)
                    else:
                        if len(lines) == 1:
                            # write the last line of the last file.
                            # hold back the first line of this file,
                            # and treat it as the last line
                            outfile.write(last_line)
                        else:
                            # write the last line of the last file, first
                            # line of this file
                            outfile.writelines([last_line, first_line])

                    # write the bulk of the lines
                    outfile.writelines(lines[1:-1])

                    # set last_line
                    last_line = lines[-1]
                    last_row, last_coords = parse_bed4(last_line)
                    (chrom, start, end, seg) = last_coords
                    last_vals = (chrom, end, seg)
                    last_start = start

            # write the very last line of all files
            outfile.write(last_line)

    def __call__(self, world):
        self.concatenate(world)

        bed_filename = self.make_filename(self.bed_filename, world)
        layer(bed_filename, make_layer_filename(bed_filename),
              bigbed_outfilename=self.make_filename(self.bigbed_filename, world))


class PosteriorSaver(OutputSaver):
    copy_attrs = ["bedgraph_filename", "num_segs", "posterior_filenames",
                  "num_worlds"]
    header_tmpl = "track type=bedGraph name=posterior.%d \
        description=\"Segway posterior probability of label %d\" \
        visibility=dense  viewLimits=0:100 maxHeightPixels=0:0:10 \
        autoScale=off color=200,100,0 altColor=0,100,200"

    def make_header(self, num_seg):
        return self.header_tmpl % (num_seg, num_seg)

    def __call__(self, world):
        # the final bedgraph filename, not the individual posterior_filenames
        outfilename = self.make_filename(self.bedgraph_filename, world)

        for num_seg in xrange(self.num_segs):
            # values for comparison to combine adjoining segments
            last_start = None
            last_line = ""
            last_vals = (None, None, None) # (chrom, coord, seg)

            with maybe_gzip_open(outfilename % num_seg, "w") as outfile:
                # XXX: add in browser track line (see SVN revisions
                # previous to 195)
                print >>outfile, self.make_header(num_seg)

                for posterior_filename in self.posterior_filenames:
                    with open(posterior_filename % num_seg) as posterior_file:
                        lines = posterior_file.readlines()
                        first_line = lines[0]
                        first_row, first_coords = parse_bed4(first_line)
                        (chrom, start, end, seg) = first_coords

                        # write the last line and the first line, after
                        # potentially merging
                        if last_vals == (chrom, start, seg):
                            first_row[INDEX_BED_START] = last_start

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
                                outfile.write(merged_line)
                        else:
                            if len(lines) == 1:
                                # write the last line of the last file.
                                # hold back the first line of this file,
                                # and treat it as the last line
                                outfile.write(last_line)
                            else:
                                # write the last line of the last file, first
                                # line of this file
                                outfile.writelines([last_line, first_line])

                        # write the bulk of the lines
                        outfile.writelines(lines[1:-1])

                        # set last_line
                        last_line = lines[-1]
                        last_row, last_coords = parse_bed4(last_line)
                        (chrom, start, end, seg) = last_coords
                        last_vals = (chrom, end, seg)
                        last_start = start

                # write the very last line of all files
                outfile.write(last_line)
