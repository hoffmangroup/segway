#!/usr/bin/env python
from __future__ import division

"""input_master.py: write input master files
"""

__version__ = "$Revision$"

## Copyright 2012, 2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from math import frexp, ldexp
from string import Template

from genomedata._util import fill_array
from numpy import (array, empty, float32, outer, sqrt, tile, vectorize, where,
                   zeros)

from ._util import (copy_attrs, data_string, DISTRIBUTION_GAMMA,
                    DISTRIBUTION_NORM, DISTRIBUTION_ASINH_NORMAL,
                    OFFSET_END, OFFSET_START, OFFSET_STEP,
                    resource_substitute, Saver,
                    SUPERVISION_UNSUPERVISED,
                    SUPERVISION_SEMISUPERVISED,
                    SUPERVISION_SUPERVISED, USE_MFSDG)

if USE_MFSDG:
    # because tying not implemented yet
    COVAR_TIED = False
else:
    COVAR_TIED = True

ABSOLUTE_FUDGE = 0.001

# here to avoid duplication
NAME_SEGCOUNTDOWN_SEG_SEGTRANSITION = "segCountDown_seg_segTransition"

CARD_SEGTRANSITION = 3

# XXX: should be options
LEN_SEG_EXPECTED = 100000
LEN_SUBSEG_EXPECTED = 100

JITTER_ORDERS_MAGNITUDE = 5  # log10(2**5) = 1.5 decimal orders of magnitude

DISTRIBUTIONS_LIKE_NORM = frozenset([DISTRIBUTION_NORM,
                                     DISTRIBUTION_ASINH_NORMAL])


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
    range_cpt = xrange(length)
    res[range_cpt, range_cpt] = prob_self_self

    return res


def format_indexed_strs(fmt, num):
    full_fmt = fmt + "%d"
    return [full_fmt % index for index in xrange(num)]


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
                  "num_track_groups", "track_groups"]

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
                  [str(num_seg) for num_seg in xrange(num_segs - 1)] +
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
                 for seg_index in xrange(self.num_segs)]

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
        + ["len_seg_strength", "num_bases"]

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
            for seg_index in xrange(num_segs):
                seg = "seg%d" % seg_index

                for subseg_index in xrange(num_subsegs):
                    subseg = "subseg%d" % subseg_index
                    mapping = dict(seg=seg, subseg=subseg,
                                   track=head_trackname)

                    rows.append(substitute_row(mapping))

            yield "\n".join(rows)


class MeanParamSpec(ParamSpec):
    type_name = "MEAN"
    object_tmpl = "mean_${seg}_${subseg}_${track} 1 ${datum}"
    jitter_std_bound = 0.2

    copy_attrs = ParamSpec.copy_attrs \
        + ["means", "random_state", "vars"]

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


class CovarParamSpec(ParamSpec):
    type_name = "COVAR"
    object_tmpl = "covar_${seg}_${subseg}_${track} 1 ${datum}"

    copy_attrs = ParamSpec.copy_attrs + ["vars"]

    def make_data(self):
        return vstack_tile(self.vars, self.num_segs, self.num_subsegs)


class TiedCovarParamSpec(CovarParamSpec):
    object_tmpl = "covar_${track} 1 ${datum}"

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
    if USE_MFSDG:
        # dimensionality component_type name mean covar weights
        object_tmpl = "1 COMPONENT_TYPE_MISSING_FEATURE_SCALED_DIAG_GAUSSIAN" \
            " mc_${distribution}_${seg}_${subseg}_${track}" \
            " mean_${seg}_${subseg}_${track} covar_${seg}_${subseg}_${track}" \
            " matrix_weightscale_1x1"
    else:
        # dimensionality component_type name mean covar
        object_tmpl = "1 COMPONENT_TYPE_DIAG_GAUSSIAN" \
            " mc_${distribution}_${seg}_${subseg}_${track}" \
            " mean_${seg}_${subseg}_${track} covar_${track}"


class GammaMCParamSpec(MCParamSpec):
    object_tmpl = "1 COMPONENT_TYPE_GAMMA mc_gamma_${seg}_${subseg}_${track}" \
        " ${min_track} gammascale_${seg}_${subseg}_${track}" \
        " gammashape_${seg}_${subseg}_${track}"


class MXParamSpec(ParamSpec):
    type_name = "MX"
    object_tmpl = "1 mx_${seg}_${subseg}_${track} 1 dpmf_always" \
        " mc_${distribution}_${seg}_${subseg}_${track}"


class InputMasterSaver(Saver):
    resource_name = "input.master.tmpl"
    copy_attrs = ["num_bases", "num_segs", "num_subsegs",
                  "num_track_groups", "card_seg_countdown",
                  "seg_countdowns_initial", "seg_table", "distribution",
                  "len_seg_strength", "resolution", "random_state",
                  "supervision_type", "use_dinucleotide", "mins", "means",
                  "vars", "gmtk_include_filename_relative", "track_groups"]

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

        return locals()  # dict of vars set in this function
