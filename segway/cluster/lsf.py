#!/usr/bin/env python
from __future__ import division

__version__ = "$Revision$"

# Copyright 2009 Michael M. Hoffman <mmh1@washington.edu>

import sys

## native specs work the same way as SGE
from .._util import ceildiv, KB, MB
from .common import _JobTemplateFactory, make_native_spec

class JobTemplateFactory(_JobTemplateFactory):
    def make_res_req(self, mem_usage):
        return "rusage[mem=%s]" % ceildiv(mem_usage, MB)

    def make_native_spec(self):
        mem_limit_spec = ceildiv(self.mem_limit, KB)

        # bsub -R: resource requirement
        # bsub -M: per-process memory limit
        # bsub -v: hard virtual memory limit for all processes
        res_spec = make_native_spec(R=self.res_req, M=mem_limit_spec, v=mem_limit_spec)

        return " ".join([self.native_spec, res_spec])

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())