#!/usr/bin/env python
from __future__ import division

"""include.py: save include file
"""

__version__ = "$Revision$"

## Copyright 2012 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from ._util import Saver

class IncludeSaver(Saver):
    resource_name = "segway.inc.tmpl"
    copy_attrs = ["card_seg_countdown", "card_supervision_label", "max_frames",
                  "num_segs", "num_subsegs", "resolution", "ruler_scale", 
                  "supervision_label_range_size"]

    def make_mapping(self):
        num_segs = self.num_segs

        if isinstance(num_segs, slice):
            num_segs = "undefined\n#error must define CARD_SEG"

        resolution = self.resolution
        ruler_scale = self.ruler_scale
        if ruler_scale % resolution != 0:
            msg = ("resolution %d is not a divisor of ruler scale %d"
                   % (resolution, ruler_scale))
            raise ValueError(msg)
        ruler_scale_scaled = ruler_scale // resolution

        return dict(card_seg=num_segs,
                    card_subseg=self.num_subsegs,
                    card_presence=resolution+1,
                    card_segCountDown=self.card_seg_countdown,
                    card_supervisionLabel=self.card_supervision_label,
                    card_frameIndex=self.max_frames,
                    supervisionLabel_rangeSize=self.supervision_label_range_size,
                    ruler_scale=ruler_scale_scaled)

