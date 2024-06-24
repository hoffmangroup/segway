#!/usr/bin/env python

"""input_master.py: write input master files
"""

__version__ = "$Revision$"

# Copyright 2012, 2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from numpy import (array, empty, outer, set_printoptions, sqrt, tile, where)
from six.moves import range

from ._util import (data_string, DISTRIBUTION_ASINH_NORMAL, DISTRIBUTION_NORM,
                    fill_array, make_default_filename, OFFSET_END,
                    OFFSET_START, OFFSET_STEP, resource_substitute,
                    SUPERVISION_SEMISUPERVISED, SUPERVISION_UNSUPERVISED,
                    USE_MFSDG, VIRTUAL_EVIDENCE_LIST_FILENAME)
from .gmtk.input_master import (DecisionTree, DenseCPT, DeterministicCPT,
                                DiagGaussianMC, DirichletTable, DPMF,
                                InputMaster, MissingFeatureDiagGaussianMC, MX,
                                RealMat, VirtualEvidence)

# NB: Currently Segway relies on older (Numpy < 1.14) printed representations
# of scalars and vectors in the parameter output. By default in newer (> 1.14)
# versions printed output "giv[es] the shortest unique representation".
# See Numpy 1.14 release notes: https://docs.scipy.org/doc/numpy/release.html
# Under heading 'Many changes to array printing, disableable with the new
# "legacy" printing mode'
try:
    # If it is a possibility, use the older printing style
    set_printoptions(legacy="1.13")
except TypeError:
    # Otherwise ignore the attempt
    pass

if USE_MFSDG:
    # because tying not implemented yet
    COVAR_TIED = False
else:
    COVAR_TIED = True


# here to avoid duplication
NAME_SEGCOUNTDOWN_SEG_SEGTRANSITION = "segCountDown_seg_segTransition"

# define the pseudocount for training the mixture distribution weights
GAUSSIAN_MIXTURE_WEIGHTS_PSEUDOCOUNT = 100

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

INPUT_MASTER_NAME = "input.master.tmpl"


def vstack_tile(array_like, *reps):
    reps = list(reps) + [1]

    return tile(array_like, reps)


def prob_transition_from_expected_len(length):
    # formula from Meta-MEME paper, Grundy WN et al. CABIOS 13:397
    # see also Reynolds SM et al. PLoS Comput Biol 4:e1000213
    # ("duration modeling")
    return length / (1 + length)


def save_input_master(runner, input_master_filename, params_dirpath=None,
                      instance_index=None):
    """
    Save the input.master file using the GMTK API.
    """

    # Preamble
    include_filename = runner.gmtk_include_filename_relative
    card_seg = runner.num_segs
    segway_preamble = \
f"""#include "{include_filename}"

#if CARD_SEG != {card_seg}
#error Specified number of segment labels (CARD_SEG) does not match the number used for this input master file ({card_seg})
#endif

"""

    # Initialize InputMaster option
    input_master = InputMaster(preamble=segway_preamble)

    # Decision Trees (DT_IN_FILE)
    map_frameIndex_ruler_tree = data_string("map_frameIndex_ruler.dt.txt")
    input_master.dt["map_frameIndex_ruler"] = \
        DecisionTree(map_frameIndex_ruler_tree)

    map_seg_segCountDown_tree = \
        make_segCountDown_tree_spec(runner, "map_seg_segCountDown.dt.tmpl")
    input_master.dt["map_seg_segCountDown"] = \
        DecisionTree(map_seg_segCountDown_tree)

    map_segTransition_ruler_seg_segCountDown_segCountDown_tree = \
        make_segCountDown_tree_spec(runner,
                                    "map_segTransition_ruler_seg_segCountDown_segCountDown.dt.tmpl")
    input_master.dt["map_segTransition_ruler_seg_segCountDown_segCountDown"] = \
        DecisionTree(map_segTransition_ruler_seg_segCountDown_segCountDown_tree)

    map_seg_subseg_obs_tree = data_string("map_seg_subseg_obs.dt.txt")
    input_master.dt["map_seg_subseg_obs"] = \
        DecisionTree(map_seg_subseg_obs_tree)

    if runner.supervision_type == SUPERVISION_SEMISUPERVISED:
        map_supervisionLabel_seg_alwaysTrue = \
            data_string("map_supervisionLabel_seg_alwaysTrue_semisupervised.dt.txt")
        input_master.dt["map_supervisionLabel_seg_alwaysTrue"] = \
            DecisionTree(map_supervisionLabel_seg_alwaysTrue)
    else:
        assert runner.supervision_type == SUPERVISION_UNSUPERVISED

    # Name Collection (NAME_COLLECTION_IN_LINE)
    num_segs = runner.num_segs
    num_subsegs = runner.num_subsegs
    for track_index, track_group in enumerate(runner.track_groups):
        track_name = track_group[0].name
        name_collection_name = f"collection_seg_{track_name}"
        name_collection_items = []
        for seg_index in range(num_segs):
            seg_name = f"seg{seg_index}"
            for subseg_index in range(num_subsegs):
                subseg_name = f"subseg{subseg_index}"
                mx_name = f"mx_{seg_name}_{subseg_name}_{track_name}"
                name_collection_items.append(mx_name)
        # Name Collection (NAME_COLLECTION_IN_LINE)
        input_master.name_collection[name_collection_name] = \
            name_collection_items

    # Dirichlet Table (DIRICHLET_TAB_IN_FILE)
    if runner.len_seg_strength > 0:
        # Make Dirichlet table data
        probs = make_dense_cpt_segCountDown_seg_segTransition(runner)
        total_pseudocounts = runner.len_seg_strength * runner.num_bases
        divisor = runner.card_seg_countdown * runner.num_segs
        pseudocounts_per_row = total_pseudocounts / divisor
        pseudocounts = (probs * pseudocounts_per_row).astype(int)

        input_master.dirichlet["dirichlet_segCountDown_seg_segTransition"] = \
            DirichletTable(pseudocounts, keep_shape=True)

    # Deterministic CPTs (DETERMINISTIC_CPT_IN_FILE) for the unsupervised case
    input_master.deterministic_cpt.line_before = \
        "#if CARD_SUPERVISIONLABEL == -1"
    input_master.deterministic_cpt["seg_segCountDown"] = \
        DeterministicCPT(("CARD_SEG", ), "CARD_SEGCOUNTDOWN",
                         "map_seg_segCountDown")
    input_master.deterministic_cpt["frameIndex_ruler"] = \
        DeterministicCPT(("CARD_FRAMEINDEX", ), "CARD_RULER",
                         "map_frameIndex_ruler")
    input_master.deterministic_cpt["segTransition_ruler_seg_segCountDown_segCountDown"] = \
        DeterministicCPT(("CARD_SEGTRANSITION", "CARD_RULER", "CARD_SEG",
                          "CARD_SEGCOUNTDOWN"), "CARD_SEGCOUNTDOWN",
                         "map_segTransition_ruler_seg_segCountDown_segCountDown")
    input_master.deterministic_cpt["seg_seg_copy"] = \
        DeterministicCPT(("CARD_SEG", ), "CARD_SEG",
                         "internal:copyParent")
    input_master.deterministic_cpt["subseg_subseg_copy"] = \
        DeterministicCPT(("CARD_SUBSEG", ), "CARD_SUBSEG",
                         "internal:copyParent")
    input_master.deterministic_cpt.line_after = \
        "#endif"

    # Deterministic CPTs (DETERMINISTIC_CPT_IN_FILE) for the semisupervised case
    input_master.deterministic_cpt_semisupervised.line_before = \
        "#if CARD_SUPERVISIONLABEL != -1"
    input_master.deterministic_cpt_semisupervised["seg_segCountDown"] = \
        DeterministicCPT(("CARD_SEG", ), "CARD_SEGCOUNTDOWN",
                         "map_seg_segCountDown")
    input_master.deterministic_cpt_semisupervised["frameIndex_ruler"] = \
        DeterministicCPT(("CARD_FRAMEINDEX", ), "CARD_RULER",
                         "map_frameIndex_ruler")
    input_master.deterministic_cpt_semisupervised["segTransition_ruler_seg_segCountDown_segCountDown"] = \
        DeterministicCPT(("CARD_SEGTRANSITION", "CARD_RULER", "CARD_SEG",
                          "CARD_SEGCOUNTDOWN"), "CARD_SEGCOUNTDOWN",
                         "map_segTransition_ruler_seg_segCountDown_segCountDown")
    input_master.deterministic_cpt_semisupervised["seg_seg_copy"] = \
        DeterministicCPT(("CARD_SEG", ), "CARD_SEG",
                         "internal:copyParent")
    input_master.deterministic_cpt_semisupervised["subseg_subseg_copy"] = \
        DeterministicCPT(("CARD_SUBSEG", ), "CARD_SUBSEG",
                         "internal:copyParent")
    input_master.deterministic_cpt_semisupervised["supervisionLabel_seg_alwaysTrue"] = \
        DeterministicCPT(("CARD_SUPERVISIONLABEL", "CARD_SEG"),
                         "CARD_BOOLEAN",
                         "map_supervisionLabel_seg_alwaysTrue")
    input_master.deterministic_cpt_semisupervised.line_after = \
        "#endif"

    # Virtual Evidence (VE_CPT_IN_FILE)
    input_master.virtual_evidence.line_before = "#if VIRTUAL_EVIDENCE == 1"
    input_master.virtual_evidence["seg_virtualEvidence"] = \
        VirtualEvidence(num_segs, VIRTUAL_EVIDENCE_LIST_FILENAME)
    input_master.virtual_evidence.line_after = "#endif"

    # DenseCPT begins the block conditional on INPUT_PARAMS_FILENAME
    input_master.dense_cpt.line_before = "#ifndef INPUT_PARAMS_FILENAME"
    # Dense CPTs (DENSE_CPT_IN_FILE)
    input_master.dense_cpt["start_seg"] = DenseCPT.uniform_from_shape(num_segs)
    input_master.dense_cpt["seg_subseg"] = \
        DenseCPT(fill_array(1.0 / num_subsegs, (num_segs, num_subsegs)),
                 keep_shape=True)
    input_master.dense_cpt["seg_seg"] = \
        DenseCPT.uniform_from_shape(num_segs, num_segs,
                                    self_transition=0)
    input_master.dense_cpt["seg_subseg_subseg"] = \
        DenseCPT.uniform_from_shape(num_segs, num_subsegs, num_subsegs,
                                    self_transition=0)
    input_master.dense_cpt[NAME_SEGCOUNTDOWN_SEG_SEGTRANSITION] = \
        make_dense_cpt_segCountDown_seg_segTransition_cpt(runner)
    if runner.use_dinucleotide:
        input_master.dense_cpt["seg_dinucleotide"] = \
            make_dense_cpt_seg_dinucleotide_cpt()

    distribution = runner.distribution

    # Objects for normal distributions
    if distribution in DISTRIBUTIONS_LIKE_NORM:
        for component in range(runner.num_mix_components):
            mean_data = make_mean_data(runner)
            covar_data = make_covar_data(runner)
            for seg_index in range(num_segs):
                seg_name = f"seg{seg_index}"
                for subseg_index in range(num_subsegs):
                    subseg_name = f"subseg{subseg_index}"
                    for track_index, track_group in enumerate(runner.track_groups):
                        track_name = track_group[0].name

                        if runner.num_mix_components == 1:
                            component_suffix = ""
                        else:
                            component_suffix = f"_component{component}"

                        # Mean (MEAN_IN_FILE)
                        mean_name = f"mean_{seg_name}_{subseg_name}_{track_name}{component_suffix}"
                        input_master.mean[mean_name] = \
                            mean_data[seg_index, subseg_index, track_index]

                        # Covar (COVAR_IN_FILE)
                        # If COVAR_TIED, write one covar per track and component
                        if COVAR_TIED:
                            covar_name = f"covar_{track_name}{component_suffix}"
                            if seg_index == 0 and subseg_index == 0:
                                input_master.covar[covar_name] = \
                                    covar_data[seg_index, subseg_index,
                                               track_index]
                        else:  # Otherwise, write for every seg and subseg
                            covar_name = f"covar_{seg_name}_{subseg_name}_{track_name}{component_suffix}"
                            input_master.covar[covar_name] = \
                                covar_data[seg_index, subseg_index,
                                           track_index]

                        # Diag Gaussian MC with mean and covar name (MC_IN_FILE)
                        mc_name = f"mc_{distribution}_{seg_name}_{subseg_name}_{track_name}{component_suffix}"
                        if USE_MFSDG:  # Add weights to end of Gaussian
                            input_master.mc[mc_name] = \
                                MissingFeatureDiagGaussianMC(mean=mean_name,
                                                             covar=covar_name)
                        else:
                            input_master.mc[mc_name] = \
                                DiagGaussianMC(mean=mean_name, covar=covar_name)

        # RealMat (REAL_MAT_IN_FILE)
        if USE_MFSDG:
            input_master.real_mat["matrix_weightscale_1x1"] = RealMat(1.0)

    else:
        raise ValueError("distribution %s not supported" % distribution)

    # DPMF (DPMF_IN_FILE)
    if runner.num_mix_components == 1:
        input_master.dpmf["dpmf_always"] = DPMF.uniform_from_shape(1)
    else:
        for seg_index in range(num_segs):
            seg_name = f"seg{seg_index}"
            for subseg_index in range(num_subsegs):
                subseg_name = f"subseg{subseg_index}"
                for track_index, track_group in enumerate(runner.track_groups):
                    track_name = track_group[0].name

                    dpmf_name = f"dpmf_{seg_name}_{subseg_name}_{track_name}"
                    dpmf_obj = \
                        DPMF.uniform_from_shape(runner.num_mix_components)
                    dpmf_obj.set_dirichlet_pseudocount(GAUSSIAN_MIXTURE_WEIGHTS_PSEUDOCOUNT)
                    input_master.dpmf[dpmf_name] = dpmf_obj

    # Mixtures (MX_IN_FILE)
    for seg_index in range(num_segs):
        seg_name = f"seg{seg_index}"
        for subseg_index in range(num_subsegs):
            subseg_name = f"subseg{subseg_index}"
            for track_index, track_group in enumerate(runner.track_groups):
                track_name = track_group[0].name
                mx_name = f"mx_{seg_name}_{subseg_name}_{track_name}"

                if runner.num_mix_components == 1:
                    dpmf_name = "dpmf_always"
                else:
                    dpmf_name = f"dpmf_{seg_name}_{subseg_name}_{track_name}"

                mx_components = []
                for component in range(runner.num_mix_components):
                    if runner.num_mix_components == 1:
                        component_suffix = ""
                    else:
                        component_suffix = f"_component{component}"
                    mx_component_name = f"mc_{distribution}_{seg_name}_{subseg_name}_{track_name}{component_suffix}"
                    mx_components.append(mx_component_name)

                input_master.mx[mx_name] = MX(dpmf_name, mx_components)

    # Mixture collection ends the block conditional on INPUT_PARAMS_FILENAME
    input_master.mx.line_after = \
"""
#else

DENSE_CPT_IN_FILE INPUT_PARAMS_FILENAME ascii
MEAN_IN_FILE INPUT_PARAMS_FILENAME ascii
COVAR_IN_FILE INPUT_PARAMS_FILENAME ascii
DPMF_IN_FILE INPUT_PARAMS_FILENAME ascii
MC_IN_FILE INPUT_PARAMS_FILENAME ascii
MX_IN_FILE INPUT_PARAMS_FILENAME ascii

#endif
"""

    if not input_master_filename:
        input_master_filename = \
            make_default_filename(input_master_filename, params_dirpath,
                                  instance_index)

    input_master.save(input_master_filename)


def make_segCountDown_tree_spec(runner, resourcename):
    num_segs = runner.num_segs
    seg_countdowns_initial = runner.seg_countdowns_initial

    header = ([str(num_segs)] +
              [str(num_seg) for num_seg in range(num_segs - 1)] +
              ["default"])

    lines = [" ".join(header)]

    for seg_countdown_initial in seg_countdowns_initial:
        lines.append(f"    -1 {seg_countdown_initial}")

    tree = "\n".join(lines)

    return resource_substitute(resourcename)(tree=tree)


def make_dense_cpt_segCountDown_seg_segTransition(runner):  # noqa
    # first values are the ones where segCountDown = 0 therefore
    # the transitions to segTransition = 2 occur early on
    card_seg_countdown = runner.card_seg_countdown

    # by default, when segCountDown is high, never transition
    res = empty((card_seg_countdown, runner.num_segs, CARD_SEGTRANSITION))

    prob_seg_self_self, prob_seg_self_other = \
        calc_prob_transition(runner.resolution, LEN_SEG_EXPECTED)

    prob_subseg_self_self, prob_subseg_self_other = \
        calc_prob_transition(runner.resolution, LEN_SUBSEG_EXPECTED)

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
    table = runner.seg_table
    ends = table[:, OFFSET_END]
    bitmap_without_maximum = ends == 0

    # where() returns a tuple; this unpacks it
    labels_with_maximum, = where(~bitmap_without_maximum)
    labels_without_maximum, = where(bitmap_without_maximum)

    # labels without a maximum
    res[0, labels_without_maximum] = probs_allow_transition
    res[1:, labels_without_maximum] = probs_prevent_transition

    # labels with a maximum
    seg_countdowns_initial = runner.seg_countdowns_initial

    res[0, labels_with_maximum] = array([0.0, 0.0, 1.0])
    for label in labels_with_maximum:
        seg_countdown_initial = seg_countdowns_initial[label]
        minimum = table[label, OFFSET_START] // table[label, OFFSET_STEP]

        seg_countdown_allow = seg_countdown_initial - minimum + 1

        res[1:seg_countdown_allow, label] = probs_allow_transition
        res[seg_countdown_allow:, label] = probs_prevent_transition

    res = res.squeeze()  # Remove leading dimension of size 1
    return res


def make_dense_cpt_segCountDown_seg_segTransition_cpt(runner):
    probs = make_dense_cpt_segCountDown_seg_segTransition(runner)
    res = DenseCPT(probs, keep_shape=True)

    if runner.len_seg_strength > 0:
        res.set_dirichlet_table(NAME_SEGCOUNTDOWN_SEG_SEGTRANSITION)

    return res


def calc_prob_transition(resolution, length):
    """Calculate probability transition from scaled expected length.
    """
    length_scaled = length // resolution

    prob_self_self = prob_transition_from_expected_len(length_scaled)
    prob_self_other = 1.0 - prob_self_self

    return prob_self_self, prob_self_other


def make_dense_cpt_seg_dinucleotide_cpt(runner):
    dinucleotide_table = [make_dinucleotide_table_row(runner)
                          for _ in range(runner.num_segs)]
    return DenseCPT(dinucleotide_table)


def make_dinucleotide_table_row(runner):
    # simple one-parameter model
    gc = runner.random_state.uniform()
    at = 1 - gc

    a = at / 2
    c = gc / 2
    g = gc - c
    t = 1 - a - c - g

    acgt = array([a, c, g, t])

    # shape: (16,)
    return outer(acgt, acgt).ravel()


def make_mean_data(runner):
    num_segs = runner.num_segs
    num_subsegs = runner.num_subsegs
    means = runner.means  # indexed by track_index

    # maximum likelihood, adjusted by no more than 0.2*sd
    stds = sqrt(runner.vars)

    # tile the means of each track (num_segs, num_subsegs times)
    means_tiled = vstack_tile(means, num_segs, num_subsegs)
    stds_tiled = vstack_tile(stds, num_segs, num_subsegs)

    jitter_std_bound = 0.2
    noise = runner.random_state.uniform(-jitter_std_bound,
                                        jitter_std_bound, stds_tiled.shape)

    return means_tiled + (stds_tiled * noise)


def make_covar_data(runner):
    return vstack_tile(runner.vars, runner.num_segs, runner.num_subsegs)
