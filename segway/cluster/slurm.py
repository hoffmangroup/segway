#!/usr/bin/env python
from __future__ import division

# Copyright 2009, 2011, 2012 Michael M. Hoffman <mmh1@washington.edu>

from math import ceil
import sys

from optbuild import OptionBuilder

from .._util import MB
from .common import _JobTemplateFactory

NATIVE_SPEC_PROG = OptionBuilder()  # do not run

# Reference: https://github.com/natefoo/slurm-drmaa
NATIVE_SPEC_DEFAULT = dict(
    nodes=1,  # Number of nodes to run on
    ntasks_per_node=1,  # Number of tasks per node
    cpus_per_task=1,  # Number of cpus per task
    mincpus=1,  # Min cpus per node
    requeue=True,  # Allow requeue on node failure or job preemption
    share=True  # Allow job allocation to be shared on nodes
)

CLEAN_SAFE_TIME = int(300 * 0.9)  # Slurm minimum job age is 300


class JobTemplateFactory(_JobTemplateFactory):
    def __init__(self, *args, **kwargs):
        _JobTemplateFactory.__init__(self, *args, **kwargs)

    def make_res_req(self, mem_usage, tmp_usage):
        # XXX: Unused tmp_usage for tmp storage requirements
        # Required minimum memory
        res_req = make_single_res_req("mem", mem_usage)

        return res_req

    def make_native_spec(self):
        return " ".join([self.native_spec,
                         make_native_spec(**NATIVE_SPEC_DEFAULT),
                         self.res_req])


def make_native_spec(*args, **kwargs):
    return " ".join(NATIVE_SPEC_PROG.build_args(args=args, options=kwargs))


def get_job_max_query_lifetime():
    return CLEAN_SAFE_TIME


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
