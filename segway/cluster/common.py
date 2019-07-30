#!/usr/bin/env python
from __future__ import absolute_import, division

__version__ = "$Revision$"

# common stuff: needs to be in a different file from __init__ because
# it is imported by lsf.py and sge.py
# Copyright 2009, 2011, 2012, 2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

import sys
from tempfile import gettempdir

from optbuild import Mixin_NoConvertUnderscore, OptionBuilder_ShortOptWithSpace

from .._util import data_filename, KB, MB

NATIVE_SPEC_PROG = (Mixin_NoConvertUnderscore
                    + OptionBuilder_ShortOptWithSpace)() # do not run

# CLEAN_PERIOD in lsb.params, after which jobs are removed from
# mbatchd's memory default is 3600, multiplying by 0.5 for a margin of
# error
# XXX: check lsb.params for real value of CLEAN_PERIOD
CLEAN_SAFE_TIME = int(3600 * 0.9)

# guard space to prevent going over mem_requested allocation
MEM_GUARD = 10*MB

BASH_CMD = "bash"
RES_WRAPPER = "segway-wrapper.sh"

MSG_EDGE = """
Ran through entire memory usage progression without success.
For details, check error messages in %s.
See the Troubleshooting section of the Segway documentation."""

NULL_FILENAME = "/dev/null"

class JobError(RuntimeError):
    pass

class _JobTemplateFactory(object):
    # this might be overridden by a subclass
    set_template_output_error = True

    def __init__(self, template, tmp_usage, mem_usage_progression,
                 output_filename, error_filename):
        self.args = template.args
        self.native_spec = template.nativeSpecification
        self.mem_usage_progression = mem_usage_progression

        # set here so that segway.cluster.lsf can use it
        self.output_filename = output_filename
        self.error_filename = error_filename

        # only do this if the subclass doesn't do it for you
        if self.set_template_output_error:
            # format:
            # hostname:filename
            # no hostname = execution host
            # XXX: should add drmaa.const.PLACEHOLDER_WD to
            # beginning of relative paths--those that don't start with "/"
            template.outputPath = ":" + NULL_FILENAME
            template.errorPath = ":" + error_filename

        self.template = template
        self.tmp_usage = tmp_usage

    def __call__(self, trial_index):
        """
        returns a job template with the attributes set for the next
        memory progression step
        """
        res = self.template

        try:
            mem_usage = self.mem_usage_progression[trial_index]
        except IndexError:
            raise RuntimeError(MSG_EDGE % self.error_filename)

        self.mem_limit = int(mem_usage)
        self.res_req = self.make_res_req(mem_usage, self.tmp_usage)

        res.args = self.make_args()
        res.nativeSpecification = self.make_native_spec()

        return res

    def make_res_req(self, mem_usage, tmp_usage):
        """
        pure virtual function to be replaced in subclass

        should set self.res_req; return value undefined

        mem_usage: expected virtual memory use in bytes
        tmp_usage: expected temporary space use in bytes
        """
        raise NotImplementedError

    def make_args(self):
        """
        wrap args with segway-wrapper.sh
        """
        # ulimit args are in kibibytes
        mem_limit_kb = str(calc_mem_limit(self.mem_limit) // KB)
        wrapper_cmdline = [BASH_CMD, data_filename(RES_WRAPPER), mem_limit_kb,
                           gettempdir(), self.output_filename]

        return wrapper_cmdline + self.args

    def make_native_spec(self):
        """
        pure virtual function to be replaced in subclass

        should use self.res_req and return a string to go into
        template.nativeSpecification
        """
        raise NotImplementedError

    def delete_job_template(self, session):
        session.deleteJobTemplate(self.template)

def make_native_spec(*args, **kwargs):
    return " ".join(NATIVE_SPEC_PROG.build_args(args=args, options=kwargs))

def calc_mem_limit(mem_usage):
    """
    amount of memory allowed to be used (less than that requested)
    """
    return int(mem_usage - MEM_GUARD)


def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
