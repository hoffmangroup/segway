#!/bin/env python
import sys
import os
import argparse
import subprocess
import random
import gzip
import shutil
from path import path
_folder_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(_folder_path)

def parse_freqs(freqs):
    ret = map(float, freqs.split(","))
    assert (abs(sum(ret) - 1.0) < 0.01)
    return ret

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir')
    parser.add_argument('--variance', type=float, default=0.1)
    parser.add_argument('--seg-len', type=int, default=100) # XXX
    parser.add_argument('--window-len', type=int, default=100, # XXX
                        help="segs per window")
    parser.add_argument('--num-windows', type=int, default=10) # XXX
    parser.add_argument('--freqs', type=parse_freqs, default=[0.25,0.25,0.25,0.25],
                        help="comma-delimited floats")
    args = parser.parse_args()

    outdir = path(args.outdir)
    include_coords = outdir / "include-coords.bed"
    track1 = outdir / "testtrack1_clean.bedgraph"
    track2 = outdir / "testtrack2_clean.bedgraph"
    track1_noisy = outdir / "testtrack1.bedgraph"
    track2_noisy = outdir / "testtrack2.bedgraph"
    correct_seg = outdir / "correct_seg.bed.gz"
    tracknames = outdir / "tracknames.txt"
    sequence = outdir / "chr1.fa"
    genomedata = outdir / "genomedata"
    genomedata_clean = outdir / "genomedata-clean"
    ve_labels = outdir / "empty.ve_labels"
    seg_table = outdir / "seg_table.bed"

    resolution = args.seg_len
    window_len = args.window_len
    num_windows = args.num_windows
    freqs = args.freqs
    # for applying random.choice
    label_population = sum(map(lambda i: [i]*int(freqs[i]*100), range(len(freqs))), [])
    track1_means = {0:0, 1:0, 2:1, 3:1}
    track2_means = {0:0, 1:1, 2:0, 3:1}
    variance = args.variance

    include_coords_f = open(include_coords, "w")
    track1_f = open(track1, "w")
    track2_f = open(track2, "w")
    track1_noisy_f = open(track1_noisy, "w")
    track2_noisy_f = open(track2_noisy, "w")
    correct_seg_f = gzip.open(correct_seg, "w")

    windows = []
    cur = 10000
    for i in range(num_windows):
        start = cur
        end = start + resolution*window_len
        # make space so that the window and gap together
        # is 10x the size of the window
        cur = end + (resolution*window_len*9)
        windows.append((start,end))

    correct_seg_f.write("track autoScale=off description=\"segway segmentation\"\n")

    cur_pos = 0
    for window_index, (window_start, window_end) in enumerate(windows):
        include_coords_f.write("chr1\t%s\t%s\n" % (window_start, window_end))
        assert (((window_end - window_start) % resolution) == 0)
        for i in range((window_end-window_start)/resolution):
            label = random.choice(label_population)
            pos = window_start + resolution*i
            correct_seg_f.write("chr1\t%s\t%s\t%s\n" % (pos, pos+resolution, label))

            track1_val = track1_means[label]
            track2_val = track2_means[label]
            track1_f.write("chr1\t%s\t%s\t%s\n" % (pos, pos+resolution, track1_val))
            track2_f.write("chr1\t%s\t%s\t%s\n" % (pos, pos+resolution, track2_val))

            track1_val_noisy = random.normalvariate(track1_val, variance)
            track2_val_noisy = random.normalvariate(track2_val, variance)
            track1_noisy_f.write("chr1\t%s\t%s\t%s\n" % (pos, pos+resolution, track1_val_noisy))
            track2_noisy_f.write("chr1\t%s\t%s\t%s\n" % (pos, pos+resolution, track2_val_noisy))


    include_coords_f.close()
    track1_f.close()
    track2_f.close()
    track1_noisy_f.close()
    track2_noisy_f.close()
    correct_seg_f.close()

    total_len = windows[-1][1]
    with open(sequence, "w") as seq_f:
        seq_f.write(">chr1\n")
        for i in range((total_len / 50)+10):
            seq_f.write("a"*50)
            seq_f.write("\n")

    with open(tracknames, "w") as tracknames_f:
        tracknames_f.write("testtrack1\ntesttrack2")

    cmd = ["genomedata-load",
           "--sequence=%s" % sequence,
           "--track=testtrack1=%s" % track1,
           "--track=testtrack2=%s" % track2,
           "--file-mode",
           "--verbose",
           genomedata_clean]
    print " ".join(cmd)
    subprocess.check_call(cmd)

    cmd2 = ["genomedata-load",
            "--sequence=%s" % sequence,
           "--track=testtrack1=%s" % track1_noisy,
           "--track=testtrack2=%s" % track2_noisy,
           "--file-mode",
           "--verbose",
           genomedata]
    print " ".join(cmd2)
    subprocess.check_call(cmd2)

    with open(ve_labels, "w") as f: pass # create empty file

    with open(seg_table, "w") as f:
        f.write("label\tlen\n:\t1::1\n")



if __name__ == '__main__':
    main()
