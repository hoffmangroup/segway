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

from .gmtk import InputMaster, NameCollection, DenseCPT, \
    DeterministicCPT, DPMF, MC, MX, Covar, Mean, DiagGaussianMC

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


class ParamSpec(object):
    """
    base class for parameter specifications used in input.master files
    """
    type_name = None
    object_tmpl = None
    copy_attrs = ["distribution", "mins", "num_segs", "num_subsegs",
                  "num_track_groups", "track_groups", "num_mix_components",
                  "means", "vars", "num_mix_components", "random_state",
                  "tracks", "resolution", "card_seg_countdown", "seg_table",
                    "seg_countdowns_initial"]

    jitter_std_bound = 0.2
    track_names = []

    def __init__(self, saver):
        # copy all variables from saver that it copied from Runner
        # XXX: override in subclasses to only copy subset
        copy_attrs(saver, self, self.copy_attrs)
        self.track_names = []
        for track in self.tracks:
            self.track_names.append(track.name)

    def __str__(self):
        return make_spec(self.type_name, self.generate_objects())
    
    def make_data(self):
        """
        override this in subclasses
        returns: container indexed by (seg_index, subseg_index, track_index)
        """
        return None

    def get_head_track_names(self):
        """
        Return list of head track names. 
        """    
        head_track_names = []
        for group in self.track_groups:
            head_track_names.append(group[0].name)
        return head_track_names

    def generate_gmtk_obj_names(self, obj, track_names):
        """
        Generate GMTK object names for the types:
        NameCollection: "col"
        entries in NameCollection: "mx_name"
        Covar: "covar", "tied_covar"
        Mean: "mean"
        MX: "mx"
        MC: "mc_diag", "mc_gamma", "mc_missing", "gammascale"
        DPMF: "dpmf"
        :param obj: str: type of gmtk object for which names must be generated
        :param track_names: list[str]: list of track names
        :return: list[str]: list of GMTK object names 
        """
        allowed_types = ["mx", "mc_diag", "mc_gamma", "mc_missing", "mean",
                     "covar", "col", "mx_name", "dpmf", "gammascale",
                     "gammashape", "tied_covar"]
        if not obj in allowed_types:
            raise ValueError("Undefined GMTK object type: {}".format(obj))
        num_segs = self.num_segs 
        num_subsegs = self.num_subsegs
        distribution = self.distribution 
        num_mix_components = self.num_mix_components
        names = []
        if obj == "covar":
            for name in track_names:
                names.append("covar_{}".format(name))
        # todo check component suffix
        elif obj == "tied_covar":
            for name in track_names:
                names.append("covar_{}".format(name))

        elif obj == "col":
            for name in track_names:
                names.append("collection_seg_{}".format(name))

        elif obj == "mx_name":
            for name in track_names:
                for i in range(num_segs):
                    for j in range(num_subsegs):
                        line = "mx_seg{}_subseg{}_{}".format(i, j, name)
                        names.append(line)

        elif obj == "dpmf" and num_mix_components == 1:
            return ["dpmf_always"]

        else:
            for i in range(num_segs):
                for j in range(num_subsegs):
                    for name in track_names:
                        # TODO check component suffix diff
                        if obj == "mc_diag":
                            line = "mc_{}_seg{}_subseg{}_{}".format(distribution,
                                                                i, j, name)
                        # TODO

                    # if obj == "mc_gamma":
                    # covered in general name generation
                    #     line = "{}_{}_seg{}_subseg{}_{}".format(obj,
                    #                         distribution, i, j, name)

                        # TODO
                        elif obj == "mc_missing":
                            line = ""

                        else:
                            line = "{}_seg{}_subseg{}_{}".format(obj, i, j, name)
                        names.append(line)

        return names

    
    def generate_name_collection(self, track_names):
        """
        Generate string representation of NameCollection objects in input master. 
        :param: track_names: list[str]: list of track names
        """
        # generate list of collection names
        collection_names = self.generate_gmtk_obj_names(obj="col",
                                                   track_names=track_names)
        #  generate list of all names in NameCollections
        names = self.generate_gmtk_obj_names("mx_name",
                                        track_names=track_names)
        num_tracks = len(track_names)
        len_name_group = int(len(names) / num_tracks)
        # names grouped by collection
        name_groups = [names[i:i + len_name_group] for i in
                       range(0, len(names), len_name_group)]
        # create NameCollection objects and add to
        # input_master.name_collection: InlineSection
        for group_index in range(len(name_groups)):
            input_master.name_collection[collection_names[group_index]] = \
                NameCollection(name_groups[group_index])

        return input_master.name_collection.__str__()


    def make_mean_data(self):
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

    def generate_mean_objects(self, track_names):
        """
        Generate string representation of Mean objects in input master. 
        :param: track_names: list[str]: list of track names 
        """
        # generate list of names of Mean objects
        names = self.generate_gmtk_obj_names("mean",
                                        track_names=track_names)
        means = self.make_mean_data()  # array
        # dimensions of means: num_segs x num_subsegs x num_head_tracks
        # create Mean objects
        names_array = array(names).reshape((self.num_segs, self.num_subsegs, len(self.track_groups)))
        for i in range(self.num_segs):
            for j in range(self.num_subsegs):
                for k in range(len(self.track_groups)):
                    input_master.mean[names_array[i, j, k]] = Mean(means[i, j, k])
        return input_master.mean.__str__()


    def generate_covar_objects(self, track_names):
        """
        Generate string representation of Covar objects in input master.
        :param: track_names: list[str]: list of track names 
        """
        if COVAR_TIED:
            names = self.generate_gmtk_obj_names("tied_covar",
                                            track_names=track_names)
        else:
            names = self.generate_gmtk_obj_names("covar",
                                            track_names=track_names)
        covars = self.vars  # array of variance values
        # create Covar objects
        for i in range(len(names)):
            input_master.covar[names[i]] = Covar(covars[i])  # TODO index error

        return input_master.covar.__str__()

    def generate_real_mat_objects(self):
        pass

    def generate_mc_objects(self, track_names):
        """
        Generate string representation of MC objects in input master. 
        :param: track_names: list[str]: list of track names 
        """
        # if distribution is norm or asinh_norm
        if self.distribution in DISTRIBUTIONS_LIKE_NORM:
            if USE_MFSDG:
                # TODO
                option = "mc_missing"
            else:
                option = "mc_diag"
            # generate MC object names
            names = self.generate_gmtk_obj_names(option,
                                            track_names=track_names)

            covar_names = list(input_master.mc.covar) * (
                        self.num_segs * self.num_subsegs)
            # replicate covar names for iteration
            mean_names = list(input_master.mc.mean)
            # list of all mean names

            # create MC objects
            for i in range(len(names)):
                input_master.mc[names[i]] = DiagGaussianMC(mean=mean_names[i],
                                                        covar=covar_names[i])
            return input_master.mc.__str__()

        # # TODO if distribution is gamma
        # elif self.distribution == DISTRIBUTION_GAMMA:
        #     option = "mc_gamma"
        #     names = generate_gmtk_obj_names(option,
        #                                     track_names=self.track_names,
        #                                     num_segs=self.num_segs,
        #                                     num_subsegs=self.num_subsegs,
        #                                     distribution=self.distribution,
        #                                     num_mix_components=self.num_mix_components)
        #     # generate gammashape and gammascale names for MC objects
        #     gamma_scale = generate_gmtk_obj_names("gammascale",
        #                                           track_names=self.track_names,
        #                                           num_segs=self.num_segs,
        #                                           num_subsegs=self.num_subsegs,
        #                                           distribution=self.distribution,
        #                                           num_mix_components=self.num_mix_components)
        #
        #     gamma_shape = generate_gmtk_obj_names("gammashape",
        #                                           track_names=self.track_names,
        #                                           num_segs=self.num_segs,
        #                                           num_subsegs=self.num_subsegs,
        #                                           distribution=self.distribution,
        #                                           num_mix_components=self.num_mix_components)
        #     # create MC objects
        #     for i in range(len(names)):
        #         mc_obj = MC(name=names[i], dim=1, type="COMPONENT_TYPE_GAMMA",
        #                     gamma_shape=gamma_shape[i],
        #                     gamma_scale=gamma_scale[i])
        #         input_master.update(mc_obj)

    def generate_mx_objects(self, track_names):
        """Generate string representation of MX objects in input master. 
        :param: track_names: list[str]: list of track names 
        """
        # generate list of MX names
        names = self.generate_gmtk_obj_names("mx",
                                        track_names=track_names)

        mc_names = list(input_master.mc) # list of all mc names
        dpmf_names = list(input_master.dpmf) # list of all dpmf names
        multiple = int(len(names) / len(dpmf_names))
        dpmf_names *= multiple  # replicate dpmf names for iteration

        # create MX objects
        for i in range(len(names)):
            input_master.mx[names[i]] = MX(dpmf=dpmf_names[i],
                        components=mc_names[i])
        return input_master.mx.__str__()


    def generate_dpmf_objects(self, track_names):
        """Generate string representation of DPMF objects in input master. 
        :param: track_names: list[str]: list of track names 
        """
        # generate a list of dpmf names
        names = self.generate_gmtk_obj_names("dpmf",
                                        track_names=track_names)
        # if single dpmf
        if self.num_mix_components == 1:
            input_master.dpmf[names[0]] = DPMF(1.0)
        else:
            # uniform probabilities
            dpmf_values = str(round(1.0 / self.num_mix_components,
                                    ROUND_NDIGITS))
            # create dpmf objects
            for i in range(len(names)):
                input_master.dpmf[names[i]] = DPMF(dpmf_values[i])
        return input_master.dpmf.__str__()


    def generate_ve(self):
        #  TODO
        pass

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

        # see Segway paper
        probs_force_transition = array([0.0, 0.0, 1.0])

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
        res[0, labels_with_maximum] = probs_force_transition
        # res[0, labels_with_maximum] = self.probs_force_transition
        for label in labels_with_maximum:
            seg_countdown_initial = seg_countdowns_initial[label]
            minimum = table[label, OFFSET_START] // table[label, OFFSET_STEP]

            seg_countdown_allow = seg_countdown_initial - minimum + 1

            res[1:seg_countdown_allow, label] = probs_allow_transition
            res[seg_countdown_allow:, label] = probs_prevent_transition

        return res


    def generate_dense_cpt_objects(self):
        # names of dense cpts
        names = ["start_seg", "seg_subseg", "seg_seg", "seg_subseg_subseg",
                 "segCountDown_seg_segTransition"]
        num_segs = self.num_segs
        num_subsegs = self.num_subsegs

        # create required probability tables
        start_seg = fill_array(1.0 / num_segs, num_segs)
        seg_subseg = fill_array(1.0 / num_subsegs, (num_segs, num_subsegs))
        seg_seg = make_zero_diagonal_table(num_segs)
        cpt_seg = make_zero_diagonal_table(num_subsegs)
        seg_subseg_subseg = (vstack_tile(cpt_seg, num_segs, 1))
        segCountDown = self.make_dense_cpt_segCountDown_seg_segTransition()
        prob = [start_seg, seg_subseg, seg_seg, seg_subseg_subseg, segCountDown]
        # create DenseCPTs and add to input_master.dense_cpt: InlineSection
        for i in range(len(names)):
            input_master.dense_cpt[names[i]] = DenseCPT(prob[i])

        return input_master.dense_cpt.__str__()

    def make_dinucleotide_table_row(self):
        pass

    def make_seg_dinucleotide(self):
        pass


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

    def make_map_segTransition_ruler_seg_segCountDown_segCountDown_dt_spec(
            self):  # noqa
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
            yield data_string(
                "map_supervisionLabel_seg_alwaysTrue_semisupervised.dt.txt")  # noqa
        elif supervision_type == SUPERVISION_SUPERVISED:
            # XXX: does not exist yet
            yield data_string(
                "map_supervisionLabel_seg_alwaysTrue_supervised.dt.txt")  # noqa
        else:
            assert supervision_type == SUPERVISION_UNSUPERVISED


class VirtualEvidenceSpec(ParamSpec):
    type_name = "VE_CPT"

    # According to GMTK specification (tksrc/GMTK_VECPT.cc)
    # this should be of the format:
    # CPT_name num_par par_card self_card VE_CPT_FILE
    # nfs:nfloats nis:nints ... fmt:obsformat ... END
    object_tmpl = "seg_virtualEvidence 1 %s 2 %s nfs:%s nis:0 fmt:ascii END"
    copy_attrs = ParamSpec.copy_attrs + ["virtual_evidence", "num_segs"]

    def make_virtual_evidence_spec(self):
        return self.object_tmpl % (
        self.num_segs, VIRTUAL_EVIDENCE_LIST_FILENAME, self.num_segs)

    def generate_objects(self):
        yield self.make_virtual_evidence_spec()


class InputMasterSaver(Saver):
    resource_name = "input.master.tmpl"
    copy_attrs = ["num_bases", "num_segs", "num_subsegs",
                  "num_track_groups", "card_seg_countdown",
                  "seg_countdowns_initial", "seg_table", "distribution",
                  "len_seg_strength", "resolution", "random_state",
                  "supervision_type",
                  "use_dinucleotide", "mins", "means", "vars",
                  "gmtk_include_filename_relative", "track_groups",
                  "num_mix_components", "virtual_evidence", "tracks"]

    def make_mapping(self):
        # the locals of this function are used as the template mapping
        # use caution before deleting or renaming any variables
        # check that they are not used in the input.master template
        param_spec = ParamSpec(self)
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

        dense_cpt_spec = param_spec.generate_dense_cpt_objects()

        # seg_seg
        num_free_params += fullnum_subsegs * (fullnum_subsegs - 1)

        # segCountDown_seg_segTransition
        num_free_params += fullnum_subsegs
        head_track_names = param_spec.get_head_track_names()
        name_collection_spec = param_spec.generate_name_collection(head_track_names)

        distribution = self.distribution
        if distribution in DISTRIBUTIONS_LIKE_NORM:
            mean_spec = param_spec.generate_mean_objects(head_track_names)
            covar_spec = param_spec.generate_covar_objects(head_track_names)
            if USE_MFSDG:
                real_mat_spec = RealMatParamSpec(self)
            else:
                real_mat_spec = ""

            mc_spec = param_spec.generate_mc_objects(head_track_names)

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
            mc_spec = param_spec.generate_mc_objects(head_track_names)

            num_free_params += (fullnum_subsegs * 2) * num_track_groups
        else:
            raise ValueError("distribution %s not supported" % distribution)

        dpmf_spec = param_spec.generate_dpmf_objects(head_track_names)
        mx_spec = param_spec.generate_mx_objects(head_track_names)
        card_seg = num_segs
        ve_spec = VirtualEvidenceSpec(self)

        return locals()  # dict of vars set in this function
