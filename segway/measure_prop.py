import sys
from os.path import basename

from .virtual_evidence import write_virtual_evidence
from ._util import (ceildiv, Copier,
                    VIRTUAL_EVIDENCE_FULL_LIST_FILENAME,
                    VIRTUAL_EVIDENCE_WINDOW_LIST_FILENAME_TMPL,
                    VIRTUAL_EVIDENCE_OBS_FILENAME_TMPL)


class MeasurePropRunner(Copier):
    copy_attrs = ["windows", "resolution", "num_segs", "uniform_ve_dirname", "work_dirpath"]

    def __init__(self, runner):
        self.runner = runner
        self.posterior_filename = None
        Copier.__init__(self, runner)

    def run_segway_posterior(self, instance_index, round_index, params_filename):
        # need to specify: MP uniform ve files, structure file, params file
        self.runner.measure_prop_ve_dirpath = self.uniform_ve_dirname
        posterior_workdir = self.runner.make_mp_posterior_workdir(instance_index,
                                                                  round_index)
        self.posterior_filenames = [(posterior_workdir /
                                     basename(self.runner.make_posterior_filename(window_index)))
                                    for window_index in range(len(self.windows))]
        self.runner.run_identify_posterior_jobs(False, True,
                                                [], posterior_filenames,
                                                params_filename,
                                                instance_index, round_index)
        self.runner.measure_prop_ve_dirpath = \
            self.runner.make_mp_posterior_workdir(instance_index, round_index)

    def load(self, instance_index):
        print >>sys.stderr, "running load_measure_prop..."

        # make uniform VE files
        def make_obs_iter():
            ve_line = [float(1)/(self.num_segs) for i in range(self.num_segs)]
            for window_index, (world, chrom, start, end) in enumerate(self.windows):
                num_frames = ceildiv(end-start, self.resolution)
                yield (ve_line for frame_index in range(num_frames))

        write_virtual_evidence(make_obs_iter(),
                               self.uniform_ve_dirname,
                               self.windows,
                               self.num_segs)

        self.runner.measure_prop_ve_dirpath = self.uniform_ve_dirname

        #self.update(instance_index, 0)


    def update(self, instance_index, round_index, params_filename):
        print >>sys.stderr, "running update_measure_prop..."

        # 2) Run gmtkJT to get posteriors for the current model
        # This runs gmtkJT and puts the posteriors at posterior_filenames
        self.run_segway_posterior(instance_index, round_index, params_filename)

        # 3) convert gmtkJT posteriors to MP label format
        for window_index, post_fname in enumerate(posterior_filenames):
            with open(fname, "r") as post:
                # XXX what is the posterior format? (look at screen about it)
                # posterior format:
                pass

        # 4) write MP graph, trans files

        # 5) run MP

        # 6) convert MP posteriors to VE
        # XXX move this to virtual_evidence.py

        #raise NotImplementedError








