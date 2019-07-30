#!/usr/bin/env python
from __future__ import absolute_import, division

__version__ = "$Revision$"

# Copyright 2009, 2011, 2012, 2014 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from os import environ
import sys
from tempfile import gettempdir

from path import path

from .._configparser import OneSectionRawConfigParser
from .._util import ceildiv, data_filename
from .common import (BASH_CMD, CLEAN_SAFE_TIME, _JobTemplateFactory,
                     make_native_spec, NULL_FILENAME)

# use SI (not binary) prefixes. I can't find any documentation for
# this but Tim Cutts seems to assume this is how it works
KB = 10**3
MB = 10**6
GB = 10**9
TB = 10**12
PB = 10**15
EB = 10**18

SIZE_UNITS = dict((unit, globals()[unit])
                  for unit in ["KB", "MB", "GB", "TB", "PB", "EB"])

LSF_CONF_BASENAME = "lsf.conf"
LSF_CONF_FILEPATH = path(environ["LSF_ENVDIR"]) / LSF_CONF_BASENAME

# XXX: this can be more robustly handled with the LSLIB function
# ls_readconfenv() and ctypes instead
LSF_CONF = OneSectionRawConfigParser(dict(LSF_UNIT_FOR_LIMITS="KB"))
LSF_CONF.read(LSF_CONF_FILEPATH)

UNIT_FOR_LIMITS = LSF_CONF.get("LSF_UNIT_FOR_LIMITS")
DIVISOR_FOR_LIMITS = SIZE_UNITS[UNIT_FOR_LIMITS]

CORE_FILE_SIZE_LIMIT = 0
HARD_RESOURCE_MULTIPLIER = 2

# this reduces failures due to systems that are simply borked for
# arbitrary reasons
PREEXEC_CMDLINE = "/bin/true"

RES_CLEAN = "segway-clean.sh"
TEMP_DIRNAME = gettempdir()

class JobTemplateFactory(_JobTemplateFactory):
    # eliminate default overwrite behavior for DRMAA/LSF, go to append
    # which is default for DRMAA/SGE
    # we use a native spec -o and -e instead of the DRMAA interface
    set_template_output_error = False

    def make_res_req(self, mem_usage, tmp_usage):
        mem_usage_spec = ceildiv(mem_usage, DIVISOR_FOR_LIMITS)
        tmp_usage_spec = ceildiv(tmp_usage, DIVISOR_FOR_LIMITS)

        # returns a quoted string
        return ('"select[mem>%s && tmp>%s] rusage[mem=%s, tmp=%s]"'
                % (mem_usage_spec, tmp_usage_spec,
                   mem_usage_spec, tmp_usage_spec))

    def make_postexec_cmdline(self):
        assert TEMP_DIRNAME.startswith("/")

        res = [BASH_CMD, data_filename(RES_CLEAN)]
        for arg in self.args:
            if arg.startswith(TEMP_DIRNAME):
                res.append(arg)

        # returns a quoted string
        return '"%s"' % " ".join(res)

    def make_native_spec(self):
        mem_limit_spec = ceildiv(self.mem_limit, DIVISOR_FOR_LIMITS)

        # these all use the unit defined by LSF_UNIT_FOR_LIMITS

        # see Administering Platform LSF: Controlling Job Execution:
        # Runtime Resource Usage Limits: Scaling the units for
        # resource usage limits

        # bsub -R: resource requirement
        # bsub -M: per-process memory limit
        # bsub -v: hard virtual memory limit for all processes, cumulatively
        # bsub -C: core file size limit
        res_spec = make_native_spec(R=self.res_req, M=mem_limit_spec,
                                    v=mem_limit_spec,
                                    o=NULL_FILENAME,
                                    e=self.error_filename,
                                    E=PREEXEC_CMDLINE,
                                    Ep=self.make_postexec_cmdline())

        # XXX: -C does not work with DRMAA for LSF 1.03
        # wait for upstream fix:
        # https://sourceforge.net/tracker/?func=detail&aid=2882034&group_id=206321&atid=997191
        #                            C=CORE_FILE_SIZE_LIMIT)
        # XXX: need to ping this

        return " ".join([self.native_spec, res_spec])


def get_job_max_query_lifetime():
    """
    Get the maximum time in seconds a job's status can be queried (by DRMAA)
    """
    return CLEAN_SAFE_TIME


def main(args=sys.argv[1:]):
    pass


if __name__ == "__main__":
    sys.exit(main())
