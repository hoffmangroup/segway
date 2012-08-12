
import struct
from path import path
import sys

from ._util import (VIRTUAL_EVIDENCE_FULL_LIST_FILENAME,
                    VIRTUAL_EVIDENCE_WINDOW_LIST_FILENAME_TMPL,
                    VIRTUAL_EVIDENCE_OBS_FILENAME_TMPL)

# obs factory is an iterator of num_windows
# iterators of num_frames(window) floats
def write_virtual_evidence(obs_iter, ve_dirname, windows, num_segs):
    #print >>sys.stderr, "writing virtual evidence..."

    ve_dirname = path(ve_dirname)
    full_list_filename = ve_dirname / VIRTUAL_EVIDENCE_FULL_LIST_FILENAME
    window_list_filenames = [(ve_dirname /
                              VIRTUAL_EVIDENCE_WINDOW_LIST_FILENAME_TMPL % window_index)
                             for window_index in range(len(windows))]
    obs_filenames = [(ve_dirname /
                      VIRTUAL_EVIDENCE_OBS_FILENAME_TMPL % window_index)
                     for window_index in range(len(windows))]

    # write full list
    with open(full_list_filename, "w") as list_file:
        list_file.write("\n".join(obs_filenames))

    # write window lists
    for window_index in range(len(windows)):
        window_list_filename = window_list_filenames[window_index]
        obs_filename = obs_filenames[window_index]
        with open(window_list_filename, "w") as window_list_file:
            window_list_file.write(obs_filename)

    # write observations
    frame_fmt = "%sf" % num_segs
    for window_index, window_obs in enumerate(obs_iter):
        obs_filename = obs_filenames[window_index]
        #print >>sys.stderr, " ", obs_filename
        if not path(obs_filename).isfile():
            with open(obs_filename, "w") as obs_file:
                for frame_index, obs in enumerate(window_obs):
                    assert (len(obs) == num_segs)
                    obs_file.write(struct.pack(frame_fmt, *obs))


    print >>sys.stderr, "done writing virtual evidence."

