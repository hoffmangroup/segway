#!/usr/bin/env python
from __future__ import division

"""local.py: local mode instead of using cluster
"""

__version__ = "$Revision$"

## Copyright 2013 Michael M. Hoffman <mmh1@uw.edu>

from math import ceil
from resource import getrlimit, RLIMIT_STACK
from subprocess import Popen

from .._util import MB
from .common import _JobTemplateFactory, make_native_spec


class JobTemplate(object):
    """mimics a DRMAA job template object
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)  # just store everything in self.__dict__


class Job(object):
    @staticmethod
    def _open_from_template_path(path):
        filename = path.partition(":")[2]  # get the filename component only
        return open(filename, "w")

    def __init__(self, job_tmpl):
        self.job_tmpl = job_tmpl

        self.outfile = self._open_from_template_path(job_tmpl.outputPath)
        self.errfile = self._open_from_template_path(job_tmpl.errorPath)

        cmd = [job_tmpl.remoteCommand] + job_tmpl.args
        self.proc = Popen(cmd,
                          env=job_tmpl.jobEnvironment,
                          stdout=self.outfile,
                          stderr=self.errfile,
                          cwd=job_tmpl.workingDirectory)

    def poll(self):
        retcode = self.proc.poll()
        if retcode is not None:
            self._close()

        return retcode

    def kill(self):
        self.proc.terminate()
        self.proc.wait()
        self._close()

    def _close(self):
        self.outfile.close()
        self.errfile.close()


class JobInfo(object):
    terminatedSignal = "LocalJobTemplateTerminatedSignal"

    def __init__(self, retcode):
        self.resourceUsage = {"cpu": "-1", "vmem": "-1", "maxvmem": "-1"}
        if not retcode is None:
            self.hasExited = True
            self.exitStatus = retcode
            self.hasSignal = (retcode < 0)
            if self.hasSignal:
                self.exitStatus *= -1  # reverse sign
                self.terminatedSignal = self.exitStatus
            self.wasAborted = False  # this would mean it never ran
        else:
            self.hasExited = False
            self.hasSignal = False
            self.wasAborted = False
        pass


class Session(object):
    """This implmentation of Session is meant to replicate the DRMAA
    Session class, but run jobs in processes on the local machine
    rather than submit jobs to a cluster.

    Sessions are responsible for cleaning up the jobs they start
    in __exit__.

    """

    TIMEOUT_NO_WAIT = "TIMEOUT_NO_WAIT"

    def __init__(self):
        self.drmsInfo = "local_virtual_drmsInfo"
        self.next_jobid = 1
        self.jobs = {}

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        for jobid, job in self.jobs.items():
            job.kill()

    def drmsInfo(self):  # noqa
        return "local"

    def runJob(self, job_tmpl):  # noqa
        jobid = str(self.next_jobid)
        self.next_jobid += 1
        self.jobs[jobid] = Job(job_tmpl)

        return jobid

    def wait(self, jobid, timeout):
        if timeout == Session.TIMEOUT_NO_WAIT:
            job = self.jobs[jobid]
            retcode = job.poll()

            if not retcode is None:
                del self.jobs[jobid]
                return JobInfo(retcode)
            else:
                raise ExitTimeoutException()
        else:
            raise NotImplementedError("Nonzero timeout not allowed"
                                      " in local mode")

    def createJobTemplate(self):  # noqa
        return JobTemplate()


class JobState:
    """
    mimics drmaa.JobState
    """
    FAILED = "Local_JobState_FAILED"
    DONE = "Local_JobState_DONE"


class ExitTimeoutException(Exception):
    "mimics drmaa.ExitTimeoutException"


XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX what is going on here?
try:
    STACK_LIMIT = min(num for num in getrlimit(RLIMIT_STACK) if num > 0)
except ValueError:
    STACK_LIMIT = 10 * MB


class JobTemplateFactory(_JobTemplateFactory):
    def make_res_req(self, mem_usage, tmp_usage):
        return []
        #return [make_single_res_req("mem_requested", mem_usage),
                #make_single_res_req("h_vmem", self.mem_limit),
                #make_single_res_req("h_stack", STACK_LIMIT)]

    def make_native_spec(self):
        # qsub -l: resource requirement
        res_spec = make_native_spec(l=self.res_req)

        res = " ".join([self.native_spec,
                        make_native_spec(**NATIVE_SPEC_DEFAULT),
                        res_spec])

        return res

def make_single_res_req(name, mem):
    # round up to the next mebibyte
    return "%s=%dM" % (name, ceil(mem / MB))
