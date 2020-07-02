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
                  "num_track_groups", "track_groups", "num_mix_components"]

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

    def make_segnames(self):
        return format_indexed_strs("seg", self.num_segs)

    def make_subsegnames(self):
        return format_indexed_strs("subseg", self.num_subsegs)

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

    def make_data(self):
        """
        override this in subclasses

        returns: container indexed by (seg_index, subseg_index, track_index)
        """
        return None

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


class TableParamSpec(ParamSpec):
    copy_attrs = ParamSpec.copy_attrs \
        + ["resolution", "card_seg_countdown", "seg_table",
           "seg_countdowns_initial"]

    # see Segway paper
    probs_force_transition = array([0.0, 0.0, 1.0])

    def make_table_spec(self, name, table, ndim, extra_rows=[]):
        header_rows = [name, ndim]
        header_rows.extend(table.shape)

        rows = [" ".join(map(str, header_rows))]
        rows.extend(extra_rows)
        rows.extend([array2text(table), ""])

        return "\n".join(rows)

    def calc_prob_transition(self, length):
        """Calculate probability transition from scaled expected length.
        """
        length_scaled = length // self.resolution

        prob_self_self = prob_transition_from_expected_len(length_scaled)
        prob_self_other = 1.0 - prob_self_self

        return prob_self_self, prob_self_other

    def make_dense_cpt_segCountDown_seg_segTransition(self):  # noqa
        # first values are the ones where segCountDown = 0 therefore
        # the transitions to segTransition = 2 occur early on
        card_seg_countdown = self.card_seg_countdown

        # by default, when segCountDown is high, never transition
        res = empty((card_seg_countdown, self.num_segs, CARD_SEGTRANSITION))

        prob_seg_self_self, prob_seg_self_other = \
            self.calc_prob_transition(LEN_SEG_EXPECTED)

        prob_subseg_self_self, prob_subseg_self_other = \
            self.calc_prob_transition(LEN_SUBSEG_EXPECTED)

        # 0: no transition
        # 1: subseg transition (no transition when CARD_SUBSEG == 1)
        # 2: seg transition
        probs_allow_transition = \
            array([prob_seg_self_self * prob_subseg_self_self,
                   prob_seg_self_self * prob_subseg_self_other,
                   prob_seg_self_other])

        probs_prevent_transition = array([prob_subseg_self_self,
                                          prob_subseg_self_other,
                                          0.0])

        # find the labels with maximum segment lengths and those without
        table = self.seg_table
        ends = table[:, OFFSET_END]
        bitmap_without_maximum = ends == 0

        # where() returns a tuple; this unpacks it
        labels_with_maximum, = where(~bitmap_without_maximum)
        labels_without_maximum, = where(bitmap_without_maximum)

        # labels without a maximum
        res[0, labels_without_maximum] = probs_allow_transition
        res[1:, labels_without_maximum] = probs_prevent_transition

        # labels with a maximum
        seg_countdowns_initial = self.seg_countdowns_initial

        res[0, labels_with_maximum] = self.probs_force_transition
        for label in labels_with_maximum:
            seg_countdown_initial = seg_countdowns_initial[label]
            minimum = table[label, OFFSET_START] // table[label, OFFSET_STEP]

            seg_countdown_allow = seg_countdown_initial - minimum + 1

            res[1:seg_countdown_allow, label] = probs_allow_transition
            res[seg_countdown_allow:, label] = probs_prevent_transition

        return res


    @staticmethod
    def make_dirichlet_name(name):
        return "dirichlet_%s" % name


class DenseCPTParamSpec(TableParamSpec):
    type_name = "DENSE_CPT"
    copy_attrs = TableParamSpec.copy_attrs \
        + ["random_state", "len_seg_strength", "use_dinucleotide"]

    def make_table_spec(self, name, table, dirichlet=False):
        """
        if dirichlet is True, this table has a corresponding DirichletTable
        automatically generated name
        """
        ndim = table.ndim - 1  # don't include output dim

        if dirichlet:
            extra_rows = ["DirichletTable %s" % self.make_dirichlet_name(name)]
        else:
            extra_rows = []

        return TableParamSpec.make_table_spec(self, name, table, ndim,
                                              extra_rows)

    def make_empty_cpt(self):
        num_segs = self.num_segs

        return zeros((num_segs, num_segs))

    def make_dense_cpt_start_seg_spec(self):
        num_segs = self.num_segs
        cpt = fill_array(1.0 / num_segs, num_segs)

        return self.make_table_spec("start_seg", cpt)

    def make_dense_cpt_seg_subseg_spec(self):
        num_subsegs = self.num_subsegs
        cpt = fill_array(1.0 / num_subsegs, (self.num_segs, num_subsegs))

        return self.make_table_spec("seg_subseg", cpt)

    def make_dense_cpt_seg_seg_spec(self):
        cpt = make_zero_diagonal_table(self.num_segs)

        return self.make_table_spec("seg_seg", cpt)

    def make_dense_cpt_seg_subseg_subseg_spec(self):
        cpt_seg = make_zero_diagonal_table(self.num_subsegs)
        cpt = vstack_tile(cpt_seg, self.num_segs, 1)

        return self.make_table_spec("seg_subseg_subseg", cpt)

    def make_dinucleotide_table_row(self):
        # simple one-parameter model
        gc = self.random_state.uniform()
        at = 1 - gc

        a = at / 2
        c = gc / 2
        g = gc - c
        t = 1 - a - c - g

        acgt = array([a, c, g, t])

        # shape: (16,)
        return outer(acgt, acgt).ravel()

    def make_dense_cpt_seg_dinucleotide_spec(self):
        table = [self.make_dinucleotide_table_row()
                 for seg_index in range(self.num_segs)]

        return self.make_table_spec("seg_dinucleotide", table)

    def make_dense_cpt_segCountDown_seg_segTransition_spec(self):  # noqa
        cpt = self.make_dense_cpt_segCountDown_seg_segTransition()

        return self.make_table_spec(NAME_SEGCOUNTDOWN_SEG_SEGTRANSITION, cpt,
                                    dirichlet=self.len_seg_strength > 0)

    def generate_objects(self):
        yield self.make_dense_cpt_start_seg_spec()
        yield self.make_dense_cpt_seg_subseg_spec()
        yield self.make_dense_cpt_seg_seg_spec()
        yield self.make_dense_cpt_seg_subseg_subseg_spec()
        yield self.make_dense_cpt_segCountDown_seg_segTransition_spec()

        if self.use_dinucleotide:
            yield self.make_dense_cpt_seg_dinucleotide_spec()


class DirichletTabParamSpec(TableParamSpec):
    type_name = "DIRICHLET_TAB"
    copy_attrs = TableParamSpec.copy_attrs \
        + ["len_seg_strength", "num_bases", "card_seg_countdown",
           "num_mix_components"]

    def make_table_spec(self, name, table):
        dirichlet_name = self.make_dirichlet_name(name)

        return TableParamSpec.make_table_spec(self, dirichlet_name, table,
                                              table.ndim)

    def make_dirichlet_table(self):
        probs = self.make_dense_cpt_segCountDown_seg_segTransition()

        # XXX: the ratio is not exact as num_bases is not the same as
        # the number of base-base transitions. It is surely close
        # enough, though
        total_pseudocounts = self.len_seg_strength * self.num_bases
        divisor = self.card_seg_countdown * self.num_segs
        pseudocounts_per_row = total_pseudocounts / divisor

        # astype(int) means flooring the floats
        pseudocounts = (probs * pseudocounts_per_row).astype(int)

        return pseudocounts

    def generate_objects(self):
        # XXX: these called functions have confusing/duplicative names
        if self.len_seg_strength > 0:
            dirichlet_table = self.make_dirichlet_table()
            yield self.make_table_spec(NAME_SEGCOUNTDOWN_SEG_SEGTRANSITION,
                                   dirichlet_table)


class NameCollectionParamSpec(ParamSpec):
    type_name = "NAME_COLLECTION"
    header_tmpl = "collection_seg_${track} ${fullnum_subsegs}"
    row_tmpl = "mx_${seg}_${subseg}_${track}"

    def generate_objects(self):
        num_segs = self.num_segs
        num_subsegs = self.num_subsegs
        track_groups = self.track_groups

        substitute_header = Template(self.header_tmpl).substitute
        substitute_row = Template(self.row_tmpl).substitute

        fullnum_subsegs = num_segs * num_subsegs

        for track_group in track_groups:
            head_trackname = track_group[0].name

            # XXX: rename in template: track -> head_trackname
            mapping = dict(track=head_trackname,
                           fullnum_subsegs=fullnum_subsegs)

            rows = [substitute_header(mapping)]
            for seg_index in range(num_segs):
                seg = "seg%d" % seg_index

                for subseg_index in range(num_subsegs):
                    subseg = "subseg%d" % subseg_index
                    mapping = dict(seg=seg, subseg=subseg,
                                   track=head_trackname)

                    rows.append(substitute_row(mapping))

            yield "\n".join(rows)


class MeanParamSpec(ParamSpec):
    type_name = "MEAN"
    object_tmpl = "mean_${seg}_${subseg}_${track}${component_suffix} 1 ${datum}"
    jitter_std_bound = 0.2

    copy_attrs = ParamSpec.copy_attrs + ["means", "num_mix_components", "random_state", "vars"]

    def make_data(self):
        num_segs = self.num_segs
        num_subsegs = self.num_subsegs
        means = self.means  # indexed by track_index

        # maximum likelihood, adjusted by no more than 0.2*sd
        stds = sqrt(self.vars)

        # tile the means of each track (num_segs, num_subsegs times)
        means_tiled = vstack_tile(means, num_segs, num_subsegs)
        stds_tiled = vstack_tile(stds, num_segs, num_subsegs)

        jitter_std_bound = self.jitter_std_bound
        noise = self.random_state.uniform(-jitter_std_bound,
                jitter_std_bound, stds_tiled.shape)

        return means_tiled + (stds_tiled * noise)


    def generate_objects(self):
        """
        returns: iterable of strs containing gmtk parameter objects starting
        with names
        """
        substitute = Template(self.object_tmpl).substitute

        for component in range(self.num_mix_components):
            data = self.make_data()
            for mapping in self.generate_tmpl_mappings():
                track_index = mapping["track_index"]
                if self.distribution == DISTRIBUTION_GAMMA:
                    mapping["min_track"] = self.get_track_lt_min(track_index)
                if data is not None:
                    seg_index = mapping["seg_index"]
                    subseg_index = mapping["subseg_index"]
                    mapping["datum"] = data[seg_index, subseg_index, track_index]
                    mapping["track"] = mapping["track"]
                    mapping["component_suffix"] = \
                        self.get_template_component_suffix(component)

                    mapping["datum"] = mapping["datum"]
                    yield substitute(mapping)


class CovarParamSpec(ParamSpec):
    type_name = "COVAR"
    object_tmpl = "covar_${seg}_${subseg}_${track}${component_suffix} 1 ${datum}"

    copy_attrs = ParamSpec.copy_attrs + ["num_mix_components", "vars"]

    def make_data(self):
        return vstack_tile(self.vars, self.num_segs, self.num_subsegs)
    
    def generate_objects(self):
        """
        returns: iterable of strs containing gmtk parameter objects starting
        with names
        """
        substitute = Template(self.object_tmpl).substitute
        for component in range(self.num_mix_components):
            data = self.make_data()
            for mapping in self.generate_tmpl_mappings():
                track_index = mapping["track_index"]
                if self.distribution == DISTRIBUTION_GAMMA:
                    mapping["min_track"] = self.get_track_lt_min(track_index)
                if data is not None:
                    seg_index = mapping["seg_index"]
                    subseg_index = mapping["subseg_index"]
                    mapping["datum"] = data[seg_index, subseg_index, track_index]
                    mapping["track"] = mapping["track"]
                    mapping["component_suffix"] = \
                        self.get_template_component_suffix(component)

                    mapping["datum"] = mapping["datum"]
                    yield substitute(mapping)
    

class TiedCovarParamSpec(CovarParamSpec):
    object_tmpl = "covar_${track}${component_suffix} 1 ${datum}"

    def make_segnames(self):
        return ["any"]

    def make_subsegnames(self):
        return ["any"]


class RealMatParamSpec(ParamSpec):
    type_name = "REAL_MAT"

    def generate_objects(self):
        yield "matrix_weightscale_1x1 1 1 1.0"


class GammaRealMatParamSpec(RealMatParamSpec):
    scale_tmpl = "gammascale_${seg}_${subseg}_${track} 1 1 ${datum}"
    shape_tmpl = "gammashape_${seg}_${subseg}_${track} 1 1 ${datum}"

    copy_attrs = ParamSpec.copy_attrs \
        + ["means", "random_state", "vars"]

    def generate_objects(self):
        means = self.means
        vars = self.vars

        substitute_scale = Template(self.scale_tmpl).substitute
        substitute_shape = Template(self.shape_tmpl).substitute

        # random start values are equivalent to the random start
        # values of a Gaussian:
        #
        # means = scales * shapes
        # vars = shapes * scales**2
        #
        # therefore:
        scales = vars / means
        shapes = (means ** 2) / vars

        for mapping in self.generate_tmpl_mappings():
            track_index = mapping["track_index"]

            scale = jitter(scales[track_index], self.random_state)
            yield substitute_scale(dict(datum=scale, **mapping))

            shape = jitter(shapes[track_index], self.random_state)
            yield substitute_shape(dict(datum=shape, **mapping))


class MCParamSpec(ParamSpec):
    type_name = "MC"


class NormMCParamSpec(MCParamSpec):
    copy_attrs = ParamSpec.copy_attrs + ["num_mix_components"]

    if USE_MFSDG:
        # dimensionality component_type name mean covar weights
        object_tmpl = "1 COMPONENT_TYPE_MISSING_FEATURE_SCALED_DIAG_GAUSSIAN" \
            " mc_${distribution}_${seg}_${subseg}_${track}" \
            " mean_${seg}_${subseg}_${track} covar_${seg}_${subseg}_${track}" \
            " matrix_weightscale_1x1"
    else:
        # dimensionality component_type name mean covar
        object_tmpl = "1 COMPONENT_TYPE_DIAG_GAUSSIAN" \
            " mc_${distribution}_${seg}_${subseg}_${track}${component_suffix}" \
            " mean_${seg}_${subseg}_${track}${component_suffix} covar_${track}${component_suffix}"

    def generate_objects(self):
        """
        returns: iterable of strs containing gmtk parameter objects starting
        with names
        """
        substitute = Template(self.object_tmpl).substitute
        for component in range(self.num_mix_components):
            for mapping in self.generate_tmpl_mappings():
                track_index = mapping["track_index"]
                if self.distribution == DISTRIBUTION_GAMMA:
                    mapping["min_track"] = self.get_track_lt_min(track_index)
                mapping["track"] = mapping["track"]
                mapping["component_suffix"] = \
                    self.get_template_component_suffix(component)

                yield substitute(mapping)


class GammaMCParamSpec(MCParamSpec):
    object_tmpl = "1 COMPONENT_TYPE_GAMMA mc_gamma_${seg}_${subseg}_${track}" \
        " ${min_track} gammascale_${seg}_${subseg}_${track}" \
        " gammashape_${seg}_${subseg}_${track}"


class MXParamSpec(ParamSpec):
    type_name = "MX"
    def generate_objects(self):
        """
        returns: iterable of strs containing gmtk parameter objects starting
        with names
        """
        object_tmpl = "1 mx_${seg}_${subseg}_${track} ${num_mix_components} "

        # If the number of mixture components is one
        if self.num_mix_components == 1:
            # Set the dense probabily mass function containing component
            # responsibilites to be set to always 1 for 1 component
            object_tmpl += "dpmf_always"
        # Otherwise set the dense probability mass function based on number
        # of components from the GMTK DPMF definition
        else:
            object_tmpl += "dpmf_${seg}_${subseg}_${track}"

        for component in range(self.num_mix_components):
            add = " mc_${distribution}_${seg}_${subseg}_${track}%s" % (
                self.get_template_component_suffix(component))
            object_tmpl += add
        substitute = Template(object_tmpl).substitute

        data = self.make_data()
        for mapping in self.generate_tmpl_mappings():
            track_index = mapping["track_index"]
            mapping["num_mix_components"] = self.num_mix_components
            if self.distribution == DISTRIBUTION_GAMMA:
                mapping["min_track"] = self.get_track_lt_min(track_index)
            if data is not None:
                seg_index = mapping["seg_index"]
                subseg_index = mapping["subseg_index"]
                mapping["datum"] = data[seg_index, subseg_index, track_index]
            yield substitute(mapping)

class DPMFParamSpec(DenseCPTParamSpec):
    type_name = "DPMF"
    copy_attrs = ParamSpec.copy_attrs + ["num_mix_components"]

    def generate_objects(self):
        """
        returns: iterable of strs containing gmtk parameter objects starting
        with names
        """
        # If the number of mixture components is one
        if self.num_mix_components == 1:
            # Create a dense probability mass function of one value of 1
            # to fix the number of mixture components to one
            yield "dpmf_always 1 1.0"
        # Otherwise
        else:
            # Create a dense probability mass function of dirichlet constants
            # with the same amount of mixture components
            object_tmpl = "dpmf_${seg}_${subseg}_${track} ${num_mix_components} "\
                        "DirichletConst %s ${weights}" % GAUSSIAN_MIXTURE_WEIGHTS_PSEUDOCOUNT
            component_weight = str(round(1.0 / self.num_mix_components,
                                         ROUND_NDIGITS))
            weights = (" " + component_weight) * self.num_mix_components
            substitute = Template(object_tmpl).substitute
            data = self.make_data()
            for mapping in self.generate_tmpl_mappings():
                mapping["weights"] = weights
                track_index = mapping["track_index"]
                mapping["num_mix_components"] = self.num_mix_components
                if self.distribution == DISTRIBUTION_GAMMA:
                    mapping["min_track"] = self.get_track_lt_min(track_index)

                if data is not None:
                    seg_index = mapping["seg_index"]
                    subseg_index = mapping["subseg_index"]
                    mapping["datum"] = data[seg_index, subseg_index, track_index]
                yield substitute(mapping)

class VirtualEvidenceSpec(ParamSpec):
    type_name = "VE_CPT"

    # According to GMTK specification (tksrc/GMTK_VECPT.cc)
    # this should be of the format: 
    # CPT_name num_par par_card self_card VE_CPT_FILE
    # nfs:nfloats nis:nints ... fmt:obsformat ... END
    object_tmpl = "seg_virtualEvidence 1 %s 2 %s nfs:%s nis:0 fmt:ascii END"
    copy_attrs = ParamSpec.copy_attrs + ["virtual_evidence", "num_segs"]

    def make_virtual_evidence_spec(self):
        return self.object_tmpl % (self.num_segs, VIRTUAL_EVIDENCE_LIST_FILENAME, self.num_segs)

    def generate_objects(self):
        yield self.make_virtual_evidence_spec()


class InputMasterSaver(Saver):
    resource_name = "input.master.tmpl"
    copy_attrs = ["num_bases", "num_segs", "num_subsegs",
                  "num_track_groups", "card_seg_countdown",
                  "seg_countdowns_initial", "seg_table", "distribution",
                  "len_seg_strength", "resolution", "random_state", "supervision_type",
                  "use_dinucleotide", "mins", "means", "vars",
                  "gmtk_include_filename_relative", "track_groups",
                  "num_mix_components", "virtual_evidence"] 

    def make_mapping(self):
        # the locals of this function are used as the template mapping
        # use caution before deleting or renaming any variables
        # check that they are not used in the input.master template
        num_free_params = 0

        num_segs = self.num_segs
        num_subsegs = self.num_subsegs
        num_track_groups = self.num_track_groups
        fullnum_subsegs = num_segs * num_subsegs

        include_filename = self.gmtk_include_filename_relative

        dt_spec = DTParamSpec(self)

        if self.len_seg_strength > 0:
            dirichlet_spec = DirichletTabParamSpec(self)
        else:
            dirichlet_spec = ""

        dense_cpt_spec = DenseCPTParamSpec(self)

        # seg_seg
        num_free_params += fullnum_subsegs * (fullnum_subsegs - 1)

        # segCountDown_seg_segTransition
        num_free_params += fullnum_subsegs

        distribution = self.distribution
        if distribution in DISTRIBUTIONS_LIKE_NORM:
            mean_spec = MeanParamSpec(self)
            if COVAR_TIED:
                covar_spec = TiedCovarParamSpec(self)
            else:
                covar_spec = CovarParamSpec(self)

            if USE_MFSDG:
                real_mat_spec = RealMatParamSpec(self)
            else:
                real_mat_spec = ""

            mc_spec = NormMCParamSpec(self)

            if COVAR_TIED:
                num_free_params += (fullnum_subsegs + 1) * num_track_groups
            else:
                num_free_params += (fullnum_subsegs * 2) * num_track_groups
        elif distribution == DISTRIBUTION_GAMMA:
            mean_spec = ""
            covar_spec = ""

            # XXX: another option is to calculate an ML estimate for
            # the gamma distribution rather than the ML estimate for the
            # mean and converting
            real_mat_spec = GammaRealMatParamSpec(self)
            mc_spec = GammaMCParamSpec(self)

            num_free_params += (fullnum_subsegs * 2) * num_track_groups
        else:
            raise ValueError("distribution %s not supported" % distribution)

        mx_spec = MXParamSpec(self)
        name_collection_spec = NameCollectionParamSpec(self)
        card_seg = num_segs
        dpmf_spec = DPMFParamSpec(self)

        ve_spec = VirtualEvidenceSpec(self)

        return locals()  # dict of vars set in this function
