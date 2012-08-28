#!/bin/env python
import sys
import os
import argparse
from path import path
import subprocess
import shutil
import random
from uuid import uuid1 as uuid
_folder_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(_folder_path)

MP_GRAPH_EXEC = "/net/noble/vol4/noble/user/maxwl/noblesvn/encode/projects/measure-prop/src/mp_files/make_mp_files"
#MP_GRAPH_EXEC = "/net/noble/vol4/noble/user/maxwl/noblesvn/encode/projects/measure-prop/bin/make_mp_segway_graph.py"
REALISTICMP_HOME = path("/net/noble/vol2/home/maxwl/segtest/realisticmp/")
MAKE_TRACK_VALS_EXEC = REALISTICMP_HOME / "make-track-vals.py"
RUN_EXEC = REALISTICMP_HOME / "segway-wrapper.sh"

BALANCED_FREQS = [0.25,0.25,0.25,0.25]
UNBALANCED_FREQS = [0.65,0.20,0.10,0.05]
FREQS = {"balanced": BALANCED_FREQS,
         "unbalanced": UNBALANCED_FREQS}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir')
    parser.add_argument('--local', action="store_true")
    args = parser.parse_args()

    workdir = path(args.workdir)
    if not workdir.isdir():
        workdir.makedirs()

    run_all_fname = workdir / "run-all.py"
    if run_all_fname.isfile():
        shutil.move(run_all_fname, str(run_all_fname) + str(uuid()))
    shutil.copyfile(path(_folder_path) / "run-all.py", run_all_fname)

    variance_choices = [0.05, 0.25, 0.5, 1.0]
    balance_choices = ["balanced", "unbalanced"]
    neighbors_choices = [100]

    num_windows = 1
    window_len = 1000
    seg_len = 10
    train_type = "identify"
    num_instances = 5

    #################################################
    # Write data files
    #################################################

    def data_dirname(variance, balance):
        return workdir / ("data-%s-%s" % (variance, balance))

    for variance in variance_choices:
        for balance in balance_choices:
            datadir = data_dirname(variance, balance)
            freqs = ",".join(map(str, FREQS[balance]))
            if not datadir.isdir():
                datadir.makedirs()
                make_vals_cmd = [MAKE_TRACK_VALS_EXEC,
                                 datadir,
                                 "--seg-len=%s" % seg_len,
                                 "--num-windows=%s" % num_windows,
                                 "--window-len=%s" % window_len,
                                 "--freqs=%s" % freqs,
                                 "--variance=%s" % variance]
                print >>sys.stderr, " ".join(make_vals_cmd)
                with open(datadir / "out.txt", "w") as out_f:
                    subprocess.check_call(make_vals_cmd, stdout=out_f, stderr=subprocess.STDOUT)

                for neighbors in neighbors_choices:
                    zcat_cmd = ["zcat", datadir / "correct_seg.bed.gz"]
                    mp_graph_cmd = [MP_GRAPH_EXEC, "indsim",
                                    datadir / "include-coords.bed",
                                    datadir / ("mp_graph_%s" % neighbors),
                                    "4", "1", str(neighbors), "false"]
                    print >>sys.stderr, " ".join(mp_graph_cmd)
                    zcat_pid = subprocess.Popen(zcat_cmd, stdout=subprocess.PIPE)
                    subprocess.check_call(mp_graph_cmd, stdin=zcat_pid.stdout)

    #################################################
    # Make param_sets
    #################################################

    mu_choices = [0.01,0.1,1.0]
    nu_choices = [0.01,0.1,1.0]
    gamma_choices = [0.01,0.1]
    mp_iters_choices = [1]
    mp_am_iters_choices = [10]
    reuse_evidence_choices = ["True", "False"]

    params_templates = [
        #[variance_choices[0], "balanced", mu_choices, nu_choices, gamma_choices, neighbors_choices[0], 1, mp_am_iters_choices, reuse_evidence_choices],
        #[variance_choices[1], "balanced", mu_choices, 0.01, 0.1, neighbors_choices[0], 1, mp_am_iters_choices, "False"],
        [variance_choices, balance_choices, 1, 0, 0, neighbors_choices[0], 1, 5, "False"],
        [variance_choices, balance_choices, 0.1, 10.0, 1.0, neighbors_choices[0], 1, 100, "False"],
        [variance_choices[2:], balance_choices, mu_choices, nu_choices, gamma_choices, neighbors_choices[0], mp_iters_choices, mp_am_iters_choices, reuse_evidence_choices],
                       ]

    def make_params_sets_aux(template, param_sets):
        if template == []: return param_sets
        if type(template[0]) == list: # what is the pythonic way of doing this?
            ret = []
            for thing in template[0]:
                new_param_sets = [param_set + [thing] for param_set in param_sets]
                ret += make_params_sets_aux(template[1:], new_param_sets)
            return ret
        else:
            new_param_sets = [param_set + [template[0]] for param_set in param_sets]
            return make_params_sets_aux(template[1:], new_param_sets)

    param_sets = []
    for template in params_templates:
        param_sets += make_params_sets_aux(template, [[]])

    #################################################
    # Stats html
    #################################################
    def write_table_row(f, lst):
        f.write("<tr>")
        for x in lst: f.write("<td>%s</td>" % x)
        f.write("</tr>\n")

    def html_frame(src):
        return ("<iframe src=\"%s\" width=100 height=40></iframe>" % src)


    stats = workdir / "stats.html"
    if stats.isfile(): stats.remove()
    stats_f = open(stats, "w")
    stats_f.write("<html><body><table>\n")
    write_table_row(stats_f, ["variance", "balance", "mu", "nu", "gamma", "neighbors", "mp-iters", "mp-am-iters", "reuse-evidence", "diagonal-frac"])


    #################################################
    # Submit jobs
    #################################################
    #param_sets = random.sample(param_sets, 5)
    for param_set in param_sets:
        datadir = data_dirname(param_set[0], param_set[1])
        job_workdir = workdir / ("workdir-" + "-".join(map(str, param_set)))
        if job_workdir.isdir(): continue
        job_workdir.makedirs()
        out_fname = job_workdir / "out.txt"
        err_fname = job_workdir / "err.txt"
        write_table_row(stats_f, param_set + [html_frame(job_workdir.basename() + "/diagonal-frac")])
        qsub_cmd = [RUN_EXEC,
                    str(job_workdir.abspath()), str(datadir.abspath()),
                    str(param_set[2]), str(param_set[3]),
                    str(param_set[4]), str(param_set[5]),
                    str(param_set[6]), str(param_set[7]),
                    str(param_set[8]),
                    str(num_instances), train_type]
        if not args.local:
            qsub_cmd = ["qsub", "-o", out_fname,
                        "-e", err_fname, "-cwd",
                        "-l", "mem_requested=8G,longjob=TRUE",
                        "-js", "10",
                        "-V"] + qsub_cmd
        print >>sys.stderr, " ".join(qsub_cmd)
        subprocess.check_call(qsub_cmd)


    stats_f.write("\n</table></body></html>\n")
    stats_f.close()

if __name__ == '__main__':
    main()
