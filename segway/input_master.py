#!/usr/bin/env python
from __future__ import absolute_import, division
"""input_master.py: write input master files
"""

__version__ = "$Revision$"

## Copyright 2012, 2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from math import frexp, ldexp
from string import Template
import sys

from genomedata._util import fill_array
from numpy import (array, empty, float32, outer, set_printoptions, sqrt, tile,
                   vectorize, where, zeros)
from six.moves import map, range

from ._util import (copy_attrs, data_string, DISTRIBUTION_GAMMA,
                    DISTRIBUTION_NORM, DISTRIBUTION_ASINH_NORMAL,
                    OFFSET_END, OFFSET_START, OFFSET_STEP,
                    resource_substitute, Saver, SEGWAY_ENCODING,
                    SUPERVISION_UNSUPERVISED,
                    SUPERVISION_SEMISUPERVISED,
                    SUPERVISION_SUPERVISED, USE_MFSDG,
                    VIRTUAL_EVIDENCE_LIST_FILENAME)

from gen_gmtk_params import InputMaster, NameCollection, DenseCPT, \
    DeterministicCPT, DPMF, MC, MX, Covar, Mean
import gen_gmtk_params


# NB: Currently Segway relies on older (Numpy < 1.14) printed representations of
# scalars and vectors in the parameter output. By default in newer (> 1.14)
# versions printed output "giv[es] the shortest unique representation".
# See Numpy 1.14 release notes: https://docs.scipy.org/doc/numpy/release.html
# Under heading 'Many changes to array printing, disableable with the new
# "legacy" printing mode'
try:
    # If it is a possibility, use the older printing style
    set_printoptions(legacy='1.13')
except TypeError:
    # Otherwise ignore the attempt
    pass

if USE_MFSDG:
    # because tying not implemented yet
    COVAR_TIED = False
else:
    COVAR_TIED = True

ABSOLUTE_FUDGE = 0.001

# define the pseudocount for training the mixture distribution weights
GAUSSIAN_MIXTURE_WEIGHTS_PSEUDOCOUNT = 100

# here to avoid duplication
NAME_SEGCOUNTDOWN_SEG_SEGTRANSITION = "segCountDown_seg_segTransition"

CARD_SEGTRANSITION = 3

# XXX: should be options
LEN_SEG_EXPECTED = 100000
LEN_SUBSEG_EXPECTED = 100

JITTER_ORDERS_MAGNITUDE = 5  # log10(2**5) = 1.5 decimal orders of magnitude

DISTRIBUTIONS_LIKE_NORM = frozenset([DISTRIBUTION_NORM,
                                     DISTRIBUTION_ASINH_NORMAL])

# Number of digits for rounding input.master means.
# This allows consistency between Python 2 and Python 3
# TODO[PY2-EOL]: remove
ROUND_NDIGITS = 12

input_master = InputMaster()

def vstack_tile(array_like, *reps):
    reps = list(reps) + [1]

    return tile(array_like, reps)


def array2text(a):
    ndim = a.ndim
    if ndim == 1:
        return " ".join(map(str, a))
    else:
        delimiter = "\n" * (ndim - 1)
        return delimiter.join(array2text(row) for row in a)


def make_spec(name, iterable):
    """
    name: str, name of GMTK object type
    iterable: iterable of strs
    """
    items = list(iterable)

    header_lines = ["%s_IN_FILE inline" % name, str(len(items)), ""]

    indexed_items = ["%d %s" % indexed_item
                     for indexed_item in enumerate(items)]

    all_lines = header_lines + indexed_items

    # In Python 2, convert from unicode to bytes to prevent
    # __str__method from being called twice
    # Specifically in the string template standard library provided by Python
    # 2, there is a call to a string escape sequence + tuple, e.g.:
    # print("%s" % (some_string,))
    # This "some_string" has its own __str__ method called *twice* if if it is
    # a unicode string in Python 2. Python 3 does not have this issue. This
    # causes downstream issues since strings are generated often in our case
    # for random numbers. Calling __str__ twice will often cause re-iterating
    # the RNG which makes for inconsitent results between Python versions.
    if sys.version[0] == "2":
        all_lines = [line.encode(SEGWAY_ENCODING) for line in all_lines]

    return "\n".join(all_lines) + "\n"


def prob_transition_from_expected_len(length):
    # formula from Meta-MEME paper, Grundy WN et al. CABIOS 13:397
    # see also Reynolds SM et al. PLoS Comput Biol 4:e1000213
    # ("duration modeling")
    return length / (1 + length)


def make_zero_diagonal_table(length):
    if length == 1:
        return array([1.0])  # always return to self

    prob_self_self = 0.0
    prob_self_other = (1.0 - prob_self_self) / (length - 1)

    # set everywhere (diagonal to be rewritten)
    res = fill_array(prob_self_other, (length, length))

    # set diagonal
    range_cpt = range(length)
    res[range_cpt, range_cpt] = prob_self_self

    return res


def format_indexed_strs(fmt, num):
    full_fmt = fmt + "%d"
    return [full_fmt % index for index in range(num)]


def jitter_cell(cell, random_state):
    """
    adds some random noise
    """
    # get the binary exponent and subtract JITTER_ORDERS_MAGNITUDE
    # e.g. 3 * 2**10 --> 1 * 2**5
    max_noise = ldexp(1, frexp(cell)[1] - JITTER_ORDERS_MAGNITUDE)

    return cell + random_state.uniform(-max_noise, max_noise)

jitter = vectorize(jitter_cell)


class ParamSpec(object):
    """
    base class for parameter specifications used in input.master files
    """
    type_name = None
    object_tmpl = None
    copy_attrs = ["distribution", "mins", "num_segs", "num_subsegs",
                  "num_track_groups", "track_groups", "num_mix_components",
                  "means", "vars", "num_mix_components", "random_state", "tracks"]

    def __init__(self, saver):
        # copy all variables from saver that it copied from Runner
        # XXX: override in subclasses to only copy subset
        copy_attrs(saver, self, self.copy_attrs)

    def get_track_lt_min(self, track_index):
        """
        returns a value less than a minimum in a track
        """
        # XXX: refactor into a new function
        min_track = self.mins[track_index]

        # fudge the minimum by a very small amount. this is not
        # continuous, but hopefully we won't get values where it
        # matters
        # XXX: restore this after GMTK issues fixed
        # if min_track == 0.0:
        #     min_track_fudged = FUDGE_TINY
        # else:
        #     min_track_fudged = min_track - ldexp(abs(min_track), FUDGE_EP)

        # this happens for really big numbers or really small
        # numbers; you only have 7 orders of magnitude to play
        # with on a float32
        min_track_f32 = float32(min_track)

        assert min_track_f32 - float32(ABSOLUTE_FUDGE) != min_track_f32
        return min_track - ABSOLUTE_FUDGE

    def get_template_component_suffix(self, component_number):
        """Returns the subsitution for the component suffix in the GMTK model
        template. Empty if there is only one component"""
        if self.num_mix_components == 1:
            return ""
        else:
            return "_component{}".format(component_number)

    def generate_tmpl_mappings(self):
        # need segnames because in the tied covariance case, the
        # segnames are replaced by "any" (see .make_covar_spec()),
        # and only one mapping is produced
        num_subsegs = self.num_subsegs

        track_groups = self.track_groups
        num_track_groups = self.num_track_groups

        for seg_index, segname in enumerate(self.make_segnames()):
            seg_offset = num_track_groups * num_subsegs * seg_index

            for subseg_index, subsegname in enumerate(self.make_subsegnames()):
                subseg_offset = seg_offset + (num_track_groups * subseg_index)

                for track_group_index, track_group in enumerate(track_groups):
                    track_offset = subseg_offset + track_group_index
                    head_trackname = track_group[0].name

                    # XXX: change name of index to track_offset in templates
                    # XXX: change name of track_index to track_group_index
                    yield dict(seg=segname, subseg=subsegname,
                               track=head_trackname, seg_index=seg_index,
                               subseg_index=subseg_index,
                               track_index=track_group_index,
                               index=track_offset,
                               distribution=self.distribution)

    def generate_objects(self):
        """
        returns: iterable of strs containing GMTK parameter objects starting
        with names
        """
        substitute = Template(self.object_tmpl).substitute

        data = self.make_data()

        for mapping in self.generate_tmpl_mappings():
            track_index = mapping["track_index"]

            if self.distribution == DISTRIBUTION_GAMMA:
                mapping["min_track"] = self.get_track_lt_min(track_index)

            if data is not None:
                seg_index = mapping["seg_index"]
                subseg_index = mapping["subseg_index"]
                mapping["datum"] = data[seg_index, subseg_index, track_index]

            yield substitute(mapping)

    def __str__(self):
        return make_spec(self.type_name, self.generate_objects())

    def generate_name_collection(self):
        collection_names = input_master.generate_gmtk_obj_names(obj="col",
                track_names=self.tracks, num_segs=self.num_segs,
                num_subsegs=self.num_subsegs, distribution=self.distribution,
                                    num_mix_components=self.num_mix_components)
        names = input_master.generate_gmtk_obj_names("mx_name",
                track_names=self.tracks, num_segs=self.num_segs,
                num_subsegs=self.num_subsegs, distribution=self.distribution,
                                    num_mix_components=self.num_mix_components)
        num_tracks = len(self.track_groups)
        len_name_group = int(len(names) / num_tracks)
        name_groups = [names[i:i + len_name_group] for i in range(0, len(names),
                                                                  len_name_group]
        for group_index in range(len(name_groups)):
            name_col = NameCollection(collection_names[group_index],
                                      name_groups[group_index])
            input_master.update(name_col)

        return input_master.generate_name_col()


    def generate_mean_objects(self):
        names = input_master.generate_gmtk_obj_names("mean",
                track_names=self.tracks, num_segs=self.num_segs,
                num_subsegs=self.num_subsegs, distribution=self.distribution,
                                    num_mix_components=self.num_mix_components)

        for i in range(len(names)):
            mean_obj = Mean(names[i], self.covars[i])
            input_master.update(mean_obj)


    def generate_covar_objects(self):
        names = input_master.generate_gmtk_obj_names("covar",
                track_names=self.tracks, num_segs=self.num_segs,
                num_subsegs=self.num_subsegs, distribution=self.distribution,
                                    num_mix_components=self.num_mix_components)

        for i in range(len(names)):
            covar_obj = Covar(names[i], self.vars[i])
            input_master.update(covar_obj)

    def generate_mc_objects(self):
        if self.distribution in DISTRIBUTIONS_LIKE_NORM:
            if USE_MFSDG:
                # TODO
                option = "mc_missing"
            else:
                option = "mc_diag"
            names = input_master.generate_gmtk_obj_names(option,
                track_names=self.tracks, num_segs=self.num_segs,
                num_subsegs=self.num_subsegs, distribution=self.distribution,
                                    num_mix_components=self.num_mix_components)
            covars = self.vars * (self.num_segs * self.num_subsegs)

            for i in range(len(names)):
                mc_obj = MC(name=names[i], dim=1, type=gen_gmtk_params.MC_TYPE_DIAG,
                            mean=list(input_master.mean)[i], covar=covars[i])
                input_master.update(mc_obj)


        elif self.distribution == DISTRIBUTION_GAMMA:
            option = "mc_gamma"
            names = input_master.generate_gmtk_obj_names(option,
                track_names=self.tracks, num_segs=self.num_segs,
                num_subsegs=self.num_subsegs, distribution=self.distribution,
                                    num_mix_components=self.num_mix_components)
            gamma_scale = input_master.generate_gmtk_obj_names("gammascale",
                track_names=self.tracks, num_segs=self.num_segs,
                num_subsegs=self.num_subsegs, distribution=self.distribution,
                                    num_mix_components=self.num_mix_components)

            gamma_shape = input_master.generate_gmtk_obj_names("gammashape",
                track_names=self.tracks, num_segs=self.num_segs,
                num_subsegs=self.num_subsegs, distribution=self.distribution,
                                    num_mix_components=self.num_mix_components)

            for i in range(len(names)):
                mc_obj = MC(name=names[i], dim=1, type=gen_gmtk_params.MC_TYPE_GAMMA,
                            gamma_shape=gamma_shape[i], gamma_scale=gamma_scale[i])
                input_master.update(mc_obj)


    def generate_mx_objects(self):
        names = input_master.generate_gmtk_obj_names("mx",
                track_names=self.tracks, num_segs=self.num_segs,
                num_subsegs=self.num_subsegs, distribution=self.distribution,
                                num_mix_components=self.num_mix_components)

        mc_names = []
        for key in input_master.mc:
            mc_names.append(input_master.mc[key].name)

        dpmf_names = []
        for key in input_master.dpmf:
            dpmf_names.append(input_master.dpmf[key].name)

        multiple = int(len(names)/len(dpmf_names))
        dpmf_names *= multiple

        for i in range(len(names)):
            mx_obj = MX(name=names[i], dim=1, dpmf=dpmf_names[i],
                        components=mc_names[i])
            input_master.update(mx_obj)


    def generate_dpmf_objects(self):
        names = input_master.generate_gmtk_obj_names("dpmf",
                track_names=self.tracks, num_segs=self.num_segs,
                num_subsegs=self.num_subsegs, distribution=self.distribution,
                                    num_mix_components=self.num_mix_components)
        if self.num_mix_components == 1:
            dpmf_obj = DPMF(names[0], 1.0)
            input_master.update(dpmf_obj)
        else:
            dpmf_values = str(round(1.0 / self.num_mix_components,
                                    ROUND_NDIGITS))
            for i in range(len(names)):
                dpmf_obj = DPMF(names[i], dpmf_values[i])
                input_master.update(dpmf_obj)


class DTParamSpec(ParamSpec):
    type_name = "DT"
    copy_attrs = ParamSpec.copy_attrs + ["seg_countdowns_initial",
                                         "supervision_type"]

    def make_segCountDown_tree_spec(self, resourcename):  # noqa
        num_segs = self.num_segs
        seg_countdowns_initial = self.seg_countdowns_initial

        header = ([str(num_segs)] +
                  [str(num_seg) for num_seg in range(num_segs - 1)] +
                  ["default"])

        lines = [" ".join(header)]

        for seg, seg_countdown_initial in enumerate(seg_countdowns_initial):
            lines.append("    -1 %d" % seg_countdown_initial)

        tree = "\n".join(lines)

        return resource_substitute(resourcename)(tree=tree)

    def make_map_seg_segCountDown_dt_spec(self):  # noqa
        return self.make_segCountDown_tree_spec("map_seg_segCountDown.dt.tmpl")

    def make_map_segTransition_ruler_seg_segCountDown_segCountDown_dt_spec(self):  # noqa
        template_name = \
            "map_segTransition_ruler_seg_segCountDown_segCountDown.dt.tmpl"
        return self.make_segCountDown_tree_spec(template_name)

    def generate_objects(self):
        yield data_string("map_frameIndex_ruler.dt.txt")
        yield self.make_map_seg_segCountDown_dt_spec()
        yield self.make_map_segTransition_ruler_seg_segCountDown_segCountDown_dt_spec()  # noqa
        yield data_string("map_seg_subseg_obs.dt.txt")

        supervision_type = self.supervision_type
        if supervision_type == SUPERVISION_SEMISUPERVISED:
            yield data_string("map_supervisionLabel_seg_alwaysTrue_semisupervised.dt.txt")  # noqa
        elif supervision_type == SUPERVISION_SUPERVISED:
            # XXX: does not exist yet
            yield data_string("map_supervisionLabel_seg_alwaysTrue_supervised.dt.txt")  # noqa
        else:
            assert supervision_type == SUPERVISION_UNSUPERVISED


