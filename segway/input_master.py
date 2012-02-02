#!/usr/bin/env python
from __future__ import division

"""input_master.py: write input master files
"""

__version__ = "$Revision$"

## Copyright 2012 Michael M. Hoffman <mmh1@uw.edu>

from functools import partial
from math import frexp, ldexp
from string import Template

from genomedata._util import fill_array
from numpy import (array, empty, float32, outer, sqrt, tile, vectorize, where,
                   zeros)
from numpy.random import uniform

from ._util import (data_string, DISTRIBUTION_GAMMA, DISTRIBUTION_NORM,
                    DISTRIBUTION_ASINH_NORMAL,
                    OFFSET_END, OFFSET_START, OFFSET_STEP,
                    resource_substitute, save_template, SUPERVISION_UNSUPERVISED,
                    SUPERVISION_SEMISUPERVISED, SUPERVISION_SUPERVISED, USE_MFSDG)

DISTRIBUTIONS_LIKE_NORM = frozenset([DISTRIBUTION_NORM,
                                     DISTRIBUTION_ASINH_NORMAL])

if USE_MFSDG:
    # because tying not implemented yet
    COVAR_TIED = False
else:
    COVAR_TIED = True

ABSOLUTE_FUDGE = 0.001

RES_INPUT_MASTER_TMPL = "input.master.tmpl"

MEAN_TMPL = "mean_${seg}_${subseg}_${track} 1 ${rand}"

COVAR_NAME_TMPL_TIED = "covar_${track}"
COVAR_NAME_TMPL_UNTIED = "covar_${seg}_${subseg}_${track}"
COVAR_TMPL_TIED = "%s 1 ${rand}" % COVAR_NAME_TMPL_TIED
COVAR_TMPL_UNTIED = "%s 1 ${rand}" % COVAR_NAME_TMPL_UNTIED

if COVAR_TIED:
    COVAR_NAME_TMPL = COVAR_NAME_TMPL_TIED
else:
    COVAR_NAME_TMPL = COVAR_NAME_TMPL_UNTIED

GAMMASCALE_TMPL = "gammascale_${seg}_${subseg}_${track} 1 1 ${rand}"
GAMMASHAPE_TMPL = "gammashape_${seg}_${subseg}_${track} 1 1 ${rand}"

if USE_MFSDG:
    MC_NORM_TMPL = "1 COMPONENT_TYPE_MISSING_FEATURE_SCALED_DIAG_GAUSSIAN" \
        " mc_${distribution}_${seg}_${subseg}_${track}" \
        " mean_${seg}_${subseg}_${track} %s" \
        " matrix_weightscale_1x1" % COVAR_NAME_TMPL
else:
    MC_NORM_TMPL = "1 COMPONENT_TYPE_DIAG_GAUSSIAN" \
        " mc_${distribution}_${seg}_${subseg}_${track}" \
        " mean_${seg}_${subseg}_${track} covar_${track}"

MC_GAMMA_TMPL = "1 COMPONENT_TYPE_GAMMA mc_gamma_${seg}_${subseg}_${track}" \
    " ${min_track} gammascale_${seg}_${subseg}_${track}" \
    " gammashape_${seg}_${subseg}_${track}"
MC_TMPLS = {"norm": MC_NORM_TMPL,
            "gamma": MC_GAMMA_TMPL,
            "asinh_norm": MC_NORM_TMPL}

MX_TMPL = "1 mx_${seg}_${subseg}_${track} 1 dpmf_always" \
    " mc_${distribution}_${seg}_${subseg}_${track}"

NAME_COLLECTION_TMPL = "collection_seg_${track} ${fullnum_subsegs}"
NAME_COLLECTION_CONTENTS_TMPL = "mx_${seg}_${subseg}_${track}"

# here to avoid duplication
NAME_SEGCOUNTDOWN_SEG_SEGTRANSITION = "segCountDown_seg_segTransition"

CARD_SEGTRANSITION = 3

# XXX: should be options
LEN_SEG_EXPECTED = 100000
LEN_SUBSEG_EXPECTED = 100

# self->self, self->other
PROBS_FORCE_TRANSITION = array([0.0, 0.0, 1.0])

## XXX: should be options
MEAN_METHOD_UNIFORM = "uniform" # randomly pick from the range
MEAN_METHOD_ML_JITTER = "ml_jitter" # maximum likelihood, then jitter

# maximum likelihood, adjusted by no more than 0.2*sd
MEAN_METHOD_ML_JITTER_STD = "ml_jitter_std"
MEAN_METHODS = [MEAN_METHOD_UNIFORM, MEAN_METHOD_ML_JITTER,
                MEAN_METHOD_ML_JITTER_STD]
MEAN_METHOD = MEAN_METHOD_ML_JITTER_STD

COVAR_METHOD_MAX_RANGE = "max_range" # maximum range
COVAR_METHOD_ML_JITTER = "ml_jitter" # maximum likelihood, then jitter
COVAR_METHOD_ML = "ml" # maximum likelihood
COVAR_METHODS = [COVAR_METHOD_MAX_RANGE, COVAR_METHOD_ML_JITTER,
                 COVAR_METHOD_ML]
COVAR_METHOD = COVAR_METHOD_ML

JITTER_ORDERS_MAGNITUDE = 5 # log10(2**5) = 1.5 decimal orders of magnitude
JITTER_STD_BOUND = 0.2

def vstack_tile(array_like, *reps):
    reps = list(reps) + [1]

    return tile(array_like, reps)

def make_dirichlet_name(name):
    return "dirichlet_%s" % name

def array2text(a):
    ndim = a.ndim
    if ndim == 1:
        return " ".join(map(str, a))
    else:
        delimiter = "\n" * (ndim-1)
        return delimiter.join(array2text(row) for row in a)

def make_spec(name, items):
    """
    name: str, name of GMTK object type
    items: list of strs
    """
    header_lines = ["%s_IN_FILE inline" % name, str(len(items)), ""]

    indexed_items = ["%d %s" % indexed_item
                     for indexed_item in enumerate(items)]

    all_lines = header_lines + indexed_items

    return "\n".join(all_lines) + "\n"

def _make_table_spec(name, table, ndim, extra_rows=[]):
    header_rows = [name, ndim]
    header_rows.extend(table.shape)

    rows = [" ".join(map(str, header_rows))]
    rows.extend(extra_rows)
    rows.extend([array2text(table), ""])

    return "\n".join(rows)

def make_table_spec(name, table, dirichlet=False):
    """
    if dirichlet is True, this table has a corresponding DirichletTable
    automatically generated name
    """
    ndim = table.ndim - 1 # don't include output dim

    if dirichlet:
        extra_rows = ["DirichletTable %s" % make_dirichlet_name(name)]
    else:
        extra_rows = []

    return _make_table_spec(name, table, ndim, extra_rows)

def make_dirichlet_table_spec(name, table):
    dirichlet_name = make_dirichlet_name(name)

    return _make_table_spec(dirichlet_name, table, table.ndim)

# def make_dt_spec(num_tracks):
#     return make_spec("DT", ["%d seg_obs%d BINARY_DT" % (index, index)
#                             for index in xrange(num_tracks)])

def prob_transition_from_expected_len(length):
    # formula from Meta-MEME paper, Grundy WN et al. CABIOS 13:397
    # see also Reynolds SM et al. PLoS Comput Biol 4:e1000213
    # ("duration modeling")
    return length / (1 + length)

def make_zero_diagonal_table(length):
    if length == 1:
        return array([1.0]) # always return to self

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

def jitter_cell(cell):
    """
    adds some random noise
    """
    # get the binary exponent and subtract JITTER_ORDERS_MAGNITUDE
    # e.g. 3 * 2**10 --> 1 * 2**5
    max_noise = ldexp(1, frexp(cell)[1] - JITTER_ORDERS_MAGNITUDE)

    return cell + uniform(-max_noise, max_noise)

jitter = vectorize(jitter_cell)

class InputMasterSaver(object):
    _copy_attrs = ["tracknames", "num_bases", "num_segs", "num_subsegs",
                   "num_tracks", "card_seg_countdown",
                   "seg_countdowns_initial", "seg_table", "distribution",
                   "len_seg_strength", "resolution", "supervision_type",
                   "use_dinucleotide", "mins", "maxs", "means", "vars",
                   "gmtk_include_filename_relative"]

    def __init__(self, runner):
        for attr in self._copy_attrs:
            setattr(self, attr, getattr(runner, attr))

    def generate_tmpl_mappings(self, segnames=None):
        # need segnames because in the tied covariance case, the
        # segnames are replaced by "any" (see .make_covar_spec()),
        # and only one mapping is produced
        num_subsegs = self.num_subsegs

        if segnames is None:
            segnames = format_indexed_strs("seg", self.num_segs)
            subsegnames = format_indexed_strs("subseg", num_subsegs)
        elif segnames == ["any"]:
            subsegnames = ["any"]

        tracknames = self.tracknames

        num_tracks = len(tracknames)

        for seg_index, segname in enumerate(segnames):
            seg_offset = num_tracks * num_subsegs * seg_index

            for subseg_index, subsegname in enumerate(subsegnames):
                subseg_offset = seg_offset + num_tracks * subseg_index

                for track_index, trackname in enumerate(tracknames):
                    track_offset = subseg_offset + track_index

                    # XXX: change name of index to track_offset in templates
                    yield dict(seg=segname, subseg=subsegname, track=trackname,
                               seg_index=seg_index, subseg_index=subseg_index,
                               track_index=track_index, index=track_offset,
                               distribution=self.distribution)

    def make_mean_spec(self):
        return self.make_spec_multiseg("MEAN", MEAN_TMPL, self.make_means())

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

    def make_items_multiseg(self, tmpl, data=None, segnames=None):
        substitute = Template(tmpl).substitute

        res = []
        for mapping in self.generate_tmpl_mappings(segnames):
            track_index = mapping["track_index"]

            if self.distribution == DISTRIBUTION_GAMMA:
                mapping["min_track"] = self.get_track_lt_min(track_index)

            if data is not None:
                seg_index = mapping["seg_index"]
                subseg_index = mapping["subseg_index"]
                mapping["rand"] = data[seg_index, subseg_index, track_index]

            res.append(substitute(mapping))

        return res

    def make_spec_multiseg(self, name, *args, **kwargs):
        items = self.make_items_multiseg(*args, **kwargs)

        return make_spec(name, items)

    def make_empty_cpt(self):
        num_segs = self.num_segs

        return zeros((num_segs, num_segs))

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

    def make_dirichlet_spec(self):
        dirichlet_table = self.make_dirichlet_table()
        # XXX: duplicative name
        items = [make_dirichlet_table_spec(NAME_SEGCOUNTDOWN_SEG_SEGTRANSITION,
                                           dirichlet_table)]

        return make_spec("DIRICHLET_TAB", items)

    def make_dense_cpt_start_seg_spec(self):
        num_segs = self.num_segs
        cpt = fill_array(1.0 / num_segs, num_segs)

        return make_table_spec("start_seg", cpt)

    def make_dense_cpt_seg_subseg_spec(self):
        num_subsegs = self.num_subsegs
        cpt = fill_array(1.0 / num_subsegs, (self.num_segs, num_subsegs))

        return make_table_spec("seg_subseg", cpt)

    def make_dense_cpt_seg_seg_spec(self):
        cpt = make_zero_diagonal_table(self.num_segs)

        return make_table_spec("seg_seg", cpt)

    def make_dense_cpt_seg_subseg_subseg_spec(self):
        cpt_seg = make_zero_diagonal_table(self.num_subsegs)
        cpt = vstack_tile(cpt_seg, self.num_segs, 1)

        return make_table_spec("seg_subseg_subseg", cpt)

    def make_dinucleotide_table_row(self):
        # simple one-parameter model
        gc = uniform()
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

        return make_table_spec("seg_dinucleotide", table)

    def calc_prob_transition_from_scaled_expected_len(self, length):
        length_scaled = length // self.resolution

        prob_self_self = prob_transition_from_expected_len(length_scaled)
        prob_self_other = 1.0 - prob_self_self

        return prob_self_self, prob_self_other

    def make_dense_cpt_segCountDown_seg_segTransition(self):
        # first values are the ones where segCountDown = 0 therefore
        # the transitions to segTransition = 2 occur early on
        card_seg_countdown = self.card_seg_countdown

        # by default, when segCountDown is high, never transition
        res = empty((card_seg_countdown, self.num_segs, CARD_SEGTRANSITION))

        prob_seg_self_self, prob_seg_self_other = \
            self.calc_prob_transition_from_scaled_expected_len(LEN_SEG_EXPECTED)

        prob_subseg_self_self, prob_subseg_self_other = \
            self.calc_prob_transition_from_scaled_expected_len(LEN_SUBSEG_EXPECTED)

        # 0: no transition
        # 1: subseg transition
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

        res[0, labels_with_maximum] = PROBS_FORCE_TRANSITION
        for label in labels_with_maximum:
            seg_countdown_initial = seg_countdowns_initial[label]
            minimum = table[label, OFFSET_START] // table[label, OFFSET_STEP]

            seg_countdown_allow = seg_countdown_initial - minimum + 1

            res[1:seg_countdown_allow, label] = probs_allow_transition
            res[seg_countdown_allow:, label] = probs_prevent_transition

        return res

    def make_dense_cpt_segCountDown_seg_segTransition_spec(self):
        cpt = self.make_dense_cpt_segCountDown_seg_segTransition()

        return make_table_spec(NAME_SEGCOUNTDOWN_SEG_SEGTRANSITION, cpt,
                               dirichlet=self.len_seg_strength > 0)

    def make_dense_cpt_spec(self):
        items = [self.make_dense_cpt_start_seg_spec(),
                 self.make_dense_cpt_seg_subseg_spec(),
                 self.make_dense_cpt_seg_seg_spec(),
                 self.make_dense_cpt_seg_subseg_subseg_spec(),
                 self.make_dense_cpt_segCountDown_seg_segTransition_spec()]

        if self.use_dinucleotide:
            items.append(self.make_dense_cpt_seg_dinucleotide_spec())

        return make_spec("DENSE_CPT", items)

    def rand_means(self):
        low = self.mins
        high = self.maxs
        num_segs = self.num_segs

        assert len(low) == len(high)

        # size parameter is so that we always get an array, even if it
        # has shape = (1,)

        # iterator so that we regenerate uniform for every seg
        return array([uniform(low, high, len(low))
                      for seg_index in xrange(num_segs)])

    def make_means(self):
        num_segs = self.num_segs
        num_subsegs = self.num_subsegs
        means = self.means

        if MEAN_METHOD == MEAN_METHOD_UNIFORM:
            raise NotImplementedError # XXX: need subseg dimenstion
            return self.rand_means()
        elif MEAN_METHOD == MEAN_METHOD_ML_JITTER:
            return jitter(vstack_tile(means, num_segs, num_subsegs))
        elif MEAN_METHOD == MEAN_METHOD_ML_JITTER_STD:
            stds = sqrt(self.vars)

            # tile the means of each track (num_segs, num_subsegs times)
            means_tiled = vstack_tile(means, num_segs, num_subsegs)
            stds_tiled = vstack_tile(stds, num_segs, num_subsegs)

            noise = uniform(-JITTER_STD_BOUND, JITTER_STD_BOUND,
                             stds_tiled.shape)

            return means_tiled + (stds_tiled * noise)

        raise ValueError("unsupported MEAN_METHOD")

    def make_covars(self):
        num_segs = self.num_segs
        num_subsegs = self.num_subsegs

        if COVAR_METHOD == COVAR_METHOD_MAX_RANGE:
            ranges = self.maxs - self.mins
            return vstack_tile(ranges, num_segs, num_subsegs)
        elif COVAR_METHOD == COVAR_METHOD_ML_JITTER:
            return jitter(vstack_tile(self.vars, num_segs, num_subsegs))
        elif COVAR_METHOD == COVAR_METHOD_ML:
            return vstack_tile(self.vars, num_segs, num_subsegs)

        raise ValueError("unsupported COVAR_METHOD")

    def make_covar_spec(self, tied):
        if tied:
            # see .generate_tmpl_mappings() for meaning of segnames
            segnames = ["any"]
            tmpl = COVAR_TMPL_TIED
        else:
            # None: automatically generate segnames
            segnames = None
            tmpl = COVAR_TMPL_UNTIED

        vars = self.make_covars()

        return self.make_spec_multiseg("COVAR", tmpl, vars, segnames)

    def make_items_gamma(self):
        means = self.means
        vars = self.vars

        substitute_scale = Template(GAMMASCALE_TMPL).substitute
        substitute_shape = Template(GAMMASHAPE_TMPL).substitute

        # random start values are equivalent to the random start
        # values of a Gaussian:
        #
        # means = scales * shapes
        # vars = shapes * scales**2
        #
        # therefore:
        scales = vars / means
        shapes = means**2 / vars

        res = []
        for mapping in self.generate_tmpl_mappings():
            track_index = mapping["track_index"]
            index = mapping["index"] * 2

            # factory for new dictionaries that start with mapping
            mapping_plus = partial(dict, **mapping)

            scale = jitter(scales[track_index])
            shape = jitter(shapes[track_index])

            mapping_scale = mapping_plus(rand=scale, index=index)
            res.append(substitute_scale(mapping_scale))

            mapping_shape = mapping_plus(rand=shape, index=index+1)
            res.append(substitute_shape(mapping_shape))

        return res

    def make_gamma_spec(self):
        return make_spec("REAL_MAT", self.make_items_gamma())

    def make_real_mat_spec(self):
        return make_spec("REAL_MAT", ["matrix_weightscale_1x1 1 1 1.0"])

    def make_mc_spec(self):
        return self.make_spec_multiseg("MC",
                                       MC_TMPLS[self.distribution])

    def make_mx_spec(self):
        return self.make_spec_multiseg("MX", MX_TMPL)

    def make_segCountDown_tree_spec(self, resourcename):
        num_segs = self.num_segs
        seg_countdowns_initial = self.seg_countdowns_initial

        header = ([str(num_segs)] +
                  [str(num_seg) for num_seg in xrange(num_segs-1)] +
                  ["default"])

        lines = [" ".join(header)]

        for seg, seg_countdown_initial in enumerate(seg_countdowns_initial):
            lines.append("    -1 %d" % seg_countdown_initial)

        tree = "\n".join(lines)

        return resource_substitute(resourcename)(tree=tree)

    def make_map_seg_segCountDown_dt_spec(self):
        return self.make_segCountDown_tree_spec("map_seg_segCountDown.dt.tmpl")

    def make_map_segTransition_ruler_seg_segCountDown_segCountDown_dt_spec(self):
        return self.make_segCountDown_tree_spec("map_segTransition_ruler_seg_segCountDown_segCountDown.dt.tmpl")

    def make_items_dt(self):
        res = []

        res.append(data_string("map_frameIndex_ruler.dt.txt"))
        res.append(self.make_map_seg_segCountDown_dt_spec())
        res.append(self.make_map_segTransition_ruler_seg_segCountDown_segCountDown_dt_spec())
        res.append(data_string("map_seg_subseg_obs.dt.txt"))

        supervision_type = self.supervision_type
        if supervision_type == SUPERVISION_SEMISUPERVISED:
            res.append(data_string("map_supervisionLabel_seg_alwaysTrue_semisupervised.dt.txt"))
        elif supervision_type == SUPERVISION_SUPERVISED:
             # XXX: does not exist yet
            res.append(data_string("map_supervisionLabel_seg_alwaysTrue_supervised.dt.txt"))
        else:
            assert supervision_type == SUPERVISION_UNSUPERVISED

        return res

    def make_dt_spec(self):
        return make_spec("DT", self.make_items_dt())

    def make_name_collection_spec(self):
        num_segs = self.num_segs
        num_subsegs = self.num_subsegs
        tracknames = self.tracknames

        substitute = Template(NAME_COLLECTION_TMPL).substitute
        substitute_contents = Template(NAME_COLLECTION_CONTENTS_TMPL).substitute

        items = []

        fullnum_subsegs = num_segs * num_subsegs

        for track_index, track in enumerate(tracknames):
            mapping = dict(track=track, fullnum_subsegs=fullnum_subsegs)

            contents = [substitute(mapping)]
            for seg_index in xrange(num_segs):
                seg = "seg%d" % seg_index

                for subseg_index in xrange(num_subsegs):
                    subseg = "subseg%d" % subseg_index
                    mapping = dict(seg=seg, subseg=subseg, track=track)

                    contents.append(substitute_contents(mapping))

            items.append("\n".join(contents))

        return make_spec("NAME_COLLECTION", items)

    def __call__(self, filename, *args, **kwargs):
        # the locals of this function are used as the template mapping
        # use caution before deleting or renaming any variables
        # check that they are not used in the input.master template
        num_free_params = 0

        num_segs = self.num_segs
        num_subsegs = self.num_subsegs
        num_tracks = self.num_tracks
        fullnum_subsegs = num_segs * num_subsegs

        include_filename = self.gmtk_include_filename_relative

        dt_spec = self.make_dt_spec()

        if self.len_seg_strength > 0:
            dirichlet_spec = self.make_dirichlet_spec()
        else:
            dirichlet_spec = ""

        dense_cpt_spec = self.make_dense_cpt_spec()

        # seg_seg
        num_free_params += fullnum_subsegs * (fullnum_subsegs - 1)

        # segCountDown_seg_segTransition
        num_free_params += fullnum_subsegs

        distribution = self.distribution
        if distribution in DISTRIBUTIONS_LIKE_NORM:
            mean_spec = self.make_mean_spec()
            covar_spec = self.make_covar_spec(COVAR_TIED)
            real_mat_spec = self.make_real_mat_spec()

            if COVAR_TIED:
                num_free_params += (fullnum_subsegs + 1) * num_tracks
            else:
                num_free_params += (fullnum_subsegs * 2) * num_tracks
        elif distribution == DISTRIBUTION_GAMMA:
            mean_spec = ""
            covar_spec = ""

            # XXX: another option is to calculate an ML estimate for
            # the gamma distribution rather than the ML estimate for the
            # mean and converting
            real_mat_spec = self.make_gamma_spec()

            num_free_params += (fullnum_subsegs * 2) * num_tracks
        else:
            raise ValueError("distribution %s not supported" % distribution)

        mc_spec = self.make_mc_spec()
        mx_spec = self.make_mx_spec()
        name_collection_spec = self.make_name_collection_spec()
        card_seg = num_segs

        return save_template(filename, RES_INPUT_MASTER_TMPL, locals(),
                             *args, **kwargs)
