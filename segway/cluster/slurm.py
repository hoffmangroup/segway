#!/usr/bin/env python
from __future__ import division

# Copyright 2009, 2011, 2012 Michael M. Hoffman <mmh1@washington.edu>

from math import ceil
import sys

from .._util import MB
from .common import _JobTemplateFactory, make_native_spec


class JobTemplateFactory(_JobTemplateFactory):

    def __init__(self, *args, **kwargs):
        _JobTemplateFactory.__init__(self, *args, **kwargs)

    def make_res_req(self, mem_usage, tmp_usage):
        # XXX: Unused tmp_usage for tmp stoarge requirements
        # Required minimum memory
        res_req = make_single_res_req("mem", mem_usage)

        return res_req

    def make_native_spec(self):
        # Reference: https://github.com/natefoo/slurm-drmaa
        native_specification = " --nodes=1 "  # Number of nodes to run on
        "--ntasks-per-node=1 "  # Number of tasks per node
        "--cpus-per-task=1 "  # Number of cpus per task
        "--mincpus=1 "  # Min cpus per node
        "--requeue "  # Allow requeue on node failure or job preemption
        "--share "  # Allow job allocation to be shared on nodes

        res_spec = make_native_spec(self.res_req)

        return res_spec + native_specification


def make_single_res_req(name, mem):
    # round up to the next mebibyte
    # http://www.clusterresources.com/torquedocs21/2.1jobsubmission.shtml#size

    # slurm native spec has form --arg=value, except when it's -a=value
    # see http://apps.man.poznan.pl/trac/slurm-drmaa
    return "--%s=%d" % (name, ceil(mem / MB))


def main(args=sys.argv[1:]):
    pass


if __name__ == "__main__":
    sys.exit(main())
