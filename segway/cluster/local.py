#!/usr/bin/env python
from __future__ import absolute_import, division

"""local.py: local mode instead of using cluster
"""

__version__ = "$Revision$"

## Copyright 2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from os import environ
from resource import getrusage, RUSAGE_CHILDREN
from subprocess import Popen
from threading import Lock
from time import sleep

from .common import (CLEAN_SAFE_TIME, _JobTemplateFactory,
                     make_native_spec)

# Allow this to throw an error if the environment variable is not an integer
MAX_PARALLEL_JOBS = int(environ.get("SEGWAY_NUM_LOCAL_JOBS", 32))
JOB_WAIT_SLEEP_TIME = 3.0

class JobTemplate(object):
    """mimics a DRMAA job template object
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)  # just store everything in self.__dict__


class Job(object):
    @staticmethod
    def _open_from_template_path(path, mode):
        filename = path.partition(":")[2]  # get the filename component only
        return open(filename, mode)

    def __init__(self, job_tmpl):
        self.job_tmpl = job_tmpl

        # Overwrite stdout of job on resubmissions
        self.outfile = self._open_from_template_path(job_tmpl.outputPath, "w")
        # Append stderr of job on resubmissions
        self.errfile = self._open_from_template_path(job_tmpl.errorPath, "a")

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
        try:
            self.proc.terminate()
            self.proc.wait()
            self._close()
        except OSError: # Ignore exception if process is no longer running
            pass

    def _close(self):
        self.outfile.close()
        self.errfile.close()


class JobInfo(object):
    terminatedSignal = "LocalJobTemplateTerminatedSignal"

    def __init__(self, retcode):
        self.resourceUsage = self._get_resource_usage()
        if retcode is not None:
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
        self.running_jobs = set() # set(jobid)

        # This lock controls access to the jobs and running_jobs objects.
        # These objects are modified in the runJob() and wait() functions.
        # runJob() holds the lock for its full extent, so that the first thread
        # to take the lock is the next to queue a job.  wait() holds the
        # lock only when it finds a finished job.
        self.lock = Lock()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        for jobid, job in self.jobs.items():
            job.kill()

    def runJob(self, job_tmpl):  # noqa
        # Wait until there are fewer than MAX_PARALLEL_JOBS running
        with self.lock:
            while len(self.running_jobs) >= MAX_PARALLEL_JOBS:
                for jobid in self.running_jobs:
                    if self.jobs[jobid].poll() is not None:
                        self.running_jobs.remove(jobid)
                        break
                else:
                    sleep(JOB_WAIT_SLEEP_TIME)

            jobid = str(self.next_jobid)
            self.next_jobid += 1
            self.jobs[jobid] = Job(job_tmpl)
            self.running_jobs.add(jobid)

        return jobid

    def wait(self, jobid, timeout):
        if timeout == Session.TIMEOUT_NO_WAIT:
            job = self.jobs[jobid]
            retcode = job.poll()

            if retcode is not None:
                with self.lock:
                    if jobid in self.jobs:
                        del self.jobs[jobid]
                    self.running_jobs.discard(jobid)
                return JobInfo(retcode)
            else:
                raise ExitTimeoutException()
        else:
            raise NotImplementedError("Nonzero timeout not allowed"
                                      " in local mode")

    def createJobTemplate(self):  # noqa
        return JobTemplate()

    def deleteJobTemplate(self, session):
        # since JobTemplate is a python object in local, upon 0
        # references it should automatically clean up after itself.
        # Thus, no deleteJobTemplate method is actually needed for
        # local.
        pass


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


def get_job_max_query_lifetime():
    """
    Get the maximum time in seconds a job's status can be queried (by DRMAA)
    """
    return CLEAN_SAFE_TIME


## here only for imports:
make_native_spec
