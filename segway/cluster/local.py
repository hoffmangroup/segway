#!/usr/bin/env python
from __future__ import division

"""local.py: local mode instead of using cluster
"""

__version__ = "$Revision$"

## Copyright 2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from resource import getrusage, RUSAGE_CHILDREN
from subprocess import Popen

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
        self.resourceUsage = self._get_resource_usage()
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

    def _get_resource_usage(self):
        # this generates results for all of the processes' children to date

        # XXX: if you want accurate results we will have to fork a
        # watcher for each sub-job

        rusage = getrusage(RUSAGE_CHILDREN)

        cpu = str(rusage.ru_utime + rusage.ru_stime)
        vmem = str(rusage.ru_ixrss + rusage.ru_idrss)

        return {"cpu": cpu, "vmem": vmem, "maxvmem": vmem}


class Session(object):
    """This implmentation of Session is meant to replicate the DRMAA
    Session class, but run jobs in processes on the local machine
    rather than submit jobs to a cluster.

    Sessions are responsible for cleaning up the jobs they start
    in __exit__.

    """

    TIMEOUT_NO_WAIT = "TIMEOUT_NO_WAIT"

    def __init__(self):
        self.drmsInfo = "local"
        self.next_jobid = 1
        self.jobs = {}

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        for jobid, job in self.jobs.items():
            job.kill()

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


class JobTemplateFactory(_JobTemplateFactory):
    def make_res_req(self, mem_usage, tmp_usage):
        return []

    def make_native_spec(self):
        return ""

## here only for imports:
make_native_spec
