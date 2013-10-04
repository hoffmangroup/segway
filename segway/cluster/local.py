#!/usr/bin/env python
from __future__ import division

__version__ = "$Revision: 8497 $"

from math import ceil
from resource import getrlimit, RLIMIT_STACK
import sys
from os import environ
from collections import namedtuple
from subprocess import Popen
from time import sleep

from .._util import MB
from .common import _JobTemplateFactory, make_native_spec

# qsub -w: switches off job validation
# qsub -j: switches off merging output and error
NATIVE_SPEC_DEFAULT = dict(w="n", j="n")

JOB_WAIT_SLEEP_TIME = 1 # seconds


# This environment variable determines how many local
# jobs should be run at once.
try:
    num_local_jobs_text = environ["SEGWAY_NUM_LOCAL_JOBS"]
    try:
        MAX_PARALLEL_JOBS = int(num_local_jobs_text)
    except ValueError:
        MAX_PARALLEL_JOBS = 32
except KeyError:
    MAX_PARALLEL_JOBS = 32

# Mimics a DRMAA job template object
class JobTemplate(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Job(object):
    def __init__(self, job_tmpl):
        self.job_tmpl = job_tmpl
        output_path = job_tmpl.outputPath.split(":")[1]
        err_path = job_tmpl.errorPath.split(":")[1]
        #self.outfd = open(output_path, "w")
        #self.errfd = open(err_path, "w")
        self.closed = False

        cmd = [job_tmpl.remoteCommand] + job_tmpl.args
        #self.proc = Popen(cmd,
                          #env=job_tmpl.jobEnvironment,
                          #stdout=self.outfd,
                          #stderr=self.errfd,
                          #cwd=job_tmpl.workingDirectory)
        self.proc = Popen(cmd,
                          env=job_tmpl.jobEnvironment,
                          cwd=job_tmpl.workingDirectory)

    def poll(self):
        retcode = self.proc.poll()
        if not retcode is None:
            self._close()
        return retcode

    #def wait(self, timeout):
        #pass

    def kill(self):
        try:
            self.proc.terminate()
        except OSError:
            # In some error conditions, the job is never created.
            # This causes proc.terminate to fail(), masking the error.
            # We'll skip throwing an exception here so that the original
            # error is printed.  In any case, leaking a process isn't
            # a big deal.
            pass
        self.proc.wait()
        self._close()

    def _close(self):
        if not self.closed:
            #self.outfd.close()
            #self.errfd.close()
            self.closed = True

class JobInfo(object):
    terminatedSignal = "LocalJobTemplateTerminatedSignal"
    def __init__(self, retcode):
        self.resourceUsage = {"cpu": "-1", "vmem": "-1", "maxvmem": "-1"}
        if not retcode is None:
            self.hasExited = True
            self.exitStatus = retcode
            self.hasSignal = (retcode < 0)
            if self.hasSignal:
                self.exitStatus *= -1
            self.wasAborted = self.hasSignal # XXX
        else:
            self.hasExited = False
            self.hasSignal = False
            self.wasAborted = False
        pass

# This implmentation of Session is meant to replicate the DRMAA Session class,
# but run jobs in processes on the local machine rather than
# submit jobs to a cluster.
#
# Sessions are responsible for cleaning up the jobs they start
# in __exit__.
class Session(object):

    TIMEOUT_NO_WAIT = "TIMEOUT_NO_WAIT"

    def __init__(self):
        self.drmsInfo = "local_virtual_drmsInfo"
        self.id_counter = 1
        self.jobs = {} # {jobid : Job}
        self.running_jobs = set() # set(jobid)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        for jobid, job in self.jobs.items():
            job.kill()

    def drmsInfo(self):
        return "local_virtual_drmsInfo"

    def runJob(self, job_tmpl):
        # Wait until there are fewere than MAX_PARALLEL_JOBS running
        while len(self.running_jobs) >= MAX_PARALLEL_JOBS:
            found_finished_job = False
            for jobid in self.running_jobs:
                if not (self.jobs[jobid].poll() is None):
                    self.running_jobs.remove(jobid)
                    found_finished_job = True
                    break
            if not found_finished_job:
                sleep(JOB_WAIT_SLEEP_TIME)

        # Start a process to run this job
        jobid = str(self.id_counter)
        self.id_counter += 1
        self.jobs[jobid] = Job(job_tmpl)
        self.running_jobs.add(jobid)
        return jobid

    def wait(self, jobid, timeout):
        if timeout == Session.TIMEOUT_NO_WAIT:
            job = self.jobs[jobid]
            retcode = job.poll()
            if not retcode is None:
                del self.jobs[jobid]
                self.running_jobs.discard(jobid)
                return JobInfo(retcode)
            else:
                raise ExitTimeoutException()
        else:
            raise NotImplementedError

    def createJobTemplate(self):
        return JobTemplate()


# Mimics drmaa.JobState
class JobState:
    FAILED = "Local_JobState_FAILED"
    DONE = "Local_JobState_DONE"

# Mimics drmaa.ExitTimeoutException
class ExitTimeoutException(Exception):
    pass


try:
    STACK_LIMIT = min(num for num in getrlimit(RLIMIT_STACK)
                      if num > 0)
except ValueError:
    STACK_LIMIT = 10*MB

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

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())

