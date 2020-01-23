#!/usr/bin/env python
from __future__ import absolute_import, division

"""structure.py: write structure file
"""

__version__ = "$Revision$"

## Copyright 2012, 2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from itertools import count

from six.moves import range, zip

from ._util import (resource_substitute, Saver, SUPERVISION_UNSUPERVISED,
                    USE_MFSDG)

MAX_WEIGHT_SCALE = 25

SUPERVISIONLABEL_WEIGHT_MULTIPLIER = 1

def add_observation(observations, resourcename, **kwargs):
    observations.append(resource_substitute(resourcename)(**kwargs))


def make_weight_scale(scale):
    return "scale %f" % scale


class StructureSaver(Saver):
    resource_name = "segway.str.tmpl"
    copy_attrs = ["num_track_groups", "num_datapoints",
                  "use_dinucleotide", "window_lens", "resolution",
                  "supervision_type", "track_groups",
                  "gmtk_include_filename_relative", "virtual_evidence",
                  "track_weight", "virtual_evidence_weight"]

    def make_weight_spec(self, multiplier):
        resolution = self.resolution
        if resolution == 1:
            return make_weight_scale(multiplier)
        else:
            # weight scale switches on the number of present data
            # points that make a model frame
            return " | ".join(make_weight_scale(index * multiplier)
                              for index in range(resolution + 1))

    def make_conditionalparents_spec(self, trackname):
        """
        this defines the parents of every observation
        """

        missing_spec = "CONDITIONALPARENTS_NIL_CONTINUOUS"
        present_spec = 'CONDITIONALPARENTS_OBS ' \
            'using mixture collection("collection_seg_%s") ' \
            'MAPPING_OBS' % trackname

        # The switching parent ("presence__...") is overloaded for
        # both switching weight and switching conditional parents. If
        # resolution > 1, then we repeat the present_spec as many
        # times as necessary to match the cardinality of the switching
        # parent
        return " | ".join([missing_spec] + [present_spec] * self.resolution)

    def add_supervision_observation(self, observation_items, next_int_track_index):
        # SUPERVISIONLABEL_WEIGHT_MULTIPLIER = 1 since we are
        # not normalising against the max length of the data tracks
        weight_spec = self.make_weight_spec(SUPERVISIONLABEL_WEIGHT_MULTIPLIER)

        # create the supervision label's conditional parents
        # using GMTK specification
        spec = 'supervisionLabel(0), seg(0) '\
            'using DeterministicCPT("supervisionLabel_seg_alwaysTrue")'
        conditionalparents_spec = " | ".join([spec] * (self.resolution+1))

        add_observation(observation_items, "supervision.tmpl",
                        track_index=next_int_track_index,
                        presence_index=next_int_track_index + 1,
                        conditionalparents_spec=conditionalparents_spec,
                        weight_spec=weight_spec)
        return

    def add_virtual_evidence_observation(self, observation_items, next_int_track_index):
        weight_spec = self.make_weight_spec(self.virtual_evidence_weight)

        # create the supervision label's conditional parents
        # using GMTK specification
        spec = 'seg(0) using VirtualEvidenceCPT("seg_virtualEvidence")'
        conditionalparents_spec = " | ".join([spec] * (self.resolution+1))

        # the index of the presence block is just 'next_int_track_index'
        # since VE only adds one int block and writes prior observations
        # in a separate file
        add_observation(observation_items, "virtual_evidence.tmpl",
                        presence_index=next_int_track_index, 
                        conditionalparents_spec=conditionalparents_spec,
                        weight_spec=weight_spec)
        return

    def make_mapping(self):
        num_track_groups = self.num_track_groups
        num_datapoints = self.num_datapoints

        assert (num_track_groups == len(num_datapoints))

        if self.use_dinucleotide:
            max_num_datapoints_track = sum(self.window_lens)
        else:
            max_num_datapoints_track = num_datapoints.max()

        observation_items = []

        zipper = zip(count(), self.track_groups, num_datapoints)
        for track_index, track_group, num_datapoints_track in zipper:
            trackname = track_group[0].name

            # relates current num_datapoints to total number of
            # possible positions. This is better than making the
            # highest num_datapoints equivalent to 1, because it
            # becomes easier to mix and match different tracks without
            # changing the weights of any of them

            # XXX: this should be done based on the minimum seg len in
            # the seg table instead
            # weight scale cannot be more than MAX_WEIGHT_SCALE to avoid
            # artifactual problems

            if self.track_weight:
                weight_multiplier = self.track_weight
            else:
                weight_multiplier = min(max_num_datapoints_track
                                        / num_datapoints_track, MAX_WEIGHT_SCALE)

            conditionalparents_spec = \
                self.make_conditionalparents_spec(trackname)

            weight_spec = self.make_weight_spec(weight_multiplier)

            # XXX: should avoid a weight line at all when weight_scale == 1.0
            # might avoid some extra multiplication in GMTK
            add_observation(observation_items, "observation.tmpl",
                            track=trackname, track_index=track_index,
                            presence_index=num_track_groups + track_index,
                            conditionalparents_spec=conditionalparents_spec,
                            weight_spec=weight_spec)

        if USE_MFSDG:
            next_int_track_index = num_track_groups + 1
        else:
            next_int_track_index = num_track_groups * 2

        # XXX: duplicative
        if self.use_dinucleotide:
            add_observation(observation_items, "dinucleotide.tmpl",
                            track_index=next_int_track_index,
                            presence_index=next_int_track_index + 1)
            next_int_track_index += 2

        if self.supervision_type != SUPERVISION_UNSUPERVISED:
            self.add_supervision_observation(observation_items, next_int_track_index)
            next_int_track_index += 2

        #if self.virtual_evidence:
        self.add_virtual_evidence_observation(observation_items, next_int_track_index)
        next_int_track_index += 1 # only adds one int block--presence data

        assert observation_items  # must be at least one track
        observations = "\n".join(observation_items)

        return dict(include_filename=self.gmtk_include_filename_relative,
                    observations=observations)
