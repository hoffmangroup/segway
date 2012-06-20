#!/bin/env python
import sys
import os
import argparse
import subprocess
import random
import gzip
_folder_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(_folder_path)

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('include_coords')
    #parser.add_argument('out')
    args = parser.parse_args()

    include_coords = "include-coords.bed"
    track1 = "testtrack1.bedgraph"
    track2 = "testtrack2.bedgraph"
    track1_noisy = "testtrack1_noisy.bedgraph"
    track2_noisy = "testtrack2_noisy.bedgraph"
    correct_seg = "correct_seg.bed.gz"

    num_windows = 5
    resolution = 10
    window_len = 1000
    freqs = [.65, .20, .10, .05]
    # for applying random.choice
    label_population = sum(map(lambda i: [i]*int(freqs[i]*100), range(len(freqs))), [])
    track1_means = {0:0, 1:0, 2:1, 3:1}
    track2_means = {0:0, 1:1, 2:0, 3:1}
    variance = 0.1

    include_coords_f = open(include_coords, "w")
    track1_f = open(track1, "w")
    track2_f = open(track2, "w")
    track1_noisy_f = open(track1_noisy, "w")
    track2_noisy_f = open(track2_noisy, "w")
    correct_seg_f = gzip.open(correct_seg, "w")

    cur_pos = 0
    for window in range(num_windows):
        window_start = (1+window*2)*window_len*resolution
        include_coords_f.write("chr1\t%s\t%s\n" % (window_start, window_start+window_len*resolution))
        for i in range(window_len):
            pos = window_start + i*resolution
            label = random.choice(label_population)
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

    cmd = ["genomedata-load", "--sequence=chr1.fa",
           "--track=testtrack1=%s" % track1,
           "--track=testtrack2=%s" % track2,
           "--file-mode",
           "--verbose",
           "genomedata"]
    print " ".join(cmd)
    subprocess.check_call(cmd)


    cmd2 = ["genomedata-load", "--sequence=chr1.fa",
           "--track=testtrack1=%s" % track1_noisy,
           "--track=testtrack2=%s" % track2_noisy,
           "--file-mode",
           "--verbose",
           "genomedata-noisy"]
    print " ".join(cmd2)
    subprocess.check_call(cmd2)

if __name__ == '__main__':
    main()
