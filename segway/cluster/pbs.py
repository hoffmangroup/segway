#!/usr/bin/env python
from __future__ import absolute_import, division

__version__ = "$Revision$"

# Copyright 2009, 2011, 2012 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from math import ceil
import sys

from .._util import MB
from .common import (CLEAN_SAFE_TIME, _JobTemplateFactory,
                     make_native_spec, NULL_FILENAME)

# pbs_drmaa requires that keep_completed be set on the server or queue
# level in order to keep job information and allow its inspection
# (such as for exit status)

class JobTemplateFactory(_JobTemplateFactory):
    # override outputPath and errorPath
    set_template_output_error = False

    def __init__(self, *args, **kwargs):
        _JobTemplateFactory.__init__(self, *args, **kwargs)

        # XXX: Jay says that these must be absolute paths. I think you
        # should be able to prepend drmaa.const.PLACEHOLDER_WD to the
        # path to get it to work, but I can't test this
        self.template.outputPath = NULL_FILENAME
        self.template.errorPath = self.error_filename.abspath()

    def make_res_req(self, mem_usage, tmp_usage):
        # XXX: is there a way to specify tmp_usage in PBS?
        return [make_single_res_req("mem", mem_usage),
                make_single_res_req("vmem", self.mem_limit)]

    def make_native_spec(self):
        # qsub -l: resource requirement
        res_spec = make_native_spec(l=self.res_req)

        res = " ".join([self.native_spec, res_spec])

        return res

def make_single_res_req(name, mem):
    # round up to the next mebibyte
    # http://www.clusterresources.com/torquedocs21/2.1jobsubmission.shtml#size
    return "%s=%dmb" % (name, ceil(mem / MB))


def get_job_max_query_lifetime():
    """
    Get the maximum time in seconds a job's status can be queried (by DRMAA)
    """
    return CLEAN_SAFE_TIME


def main(args=sys.argv[1:]):
    pass


if __name__ == "__main__":
    sys.exit(main())
