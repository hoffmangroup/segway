#!/usr/bin/env python
from __future__ import absolute_import, division

"""input_master.py: write input master files
"""

__version__ = "$Revision$"

## Copyright 2012, 2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

import sys

from genomedata._util import fill_array
from numpy import (array, empty, set_printoptions, sqrt, tile, where)
import numpy as np
from six.moves import map, range

from ._util import (copy_attrs, data_string,
                    DISTRIBUTION_NORM, DISTRIBUTION_ASINH_NORMAL,
                    OFFSET_END, OFFSET_START, OFFSET_STEP,
                    resource_substitute, Saver, SEGWAY_ENCODING,
                    SUPERVISION_UNSUPERVISED,
                    SUPERVISION_SEMISUPERVISED,
                    SUPERVISION_SUPERVISED, USE_MFSDG,
                    VIRTUAL_EVIDENCE_LIST_FILENAME)

from .gmtk.input_master import (InputMaster, NameCollection, DenseCPT,
    DPMF, MX, Covar, Mean, DiagGaussianMC)

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
                    "seg_countdowns_initial", "len_seg_strength", "num_bases"]

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
    
    def get_template_component_suffix(self, component_number):
        """Returns the subsitution for the component suffix in the GMTK model
        template. Empty if there is only one component"""
        if self.num_mix_components == 1:
            return ""
        else:
            return "_component{}".format(component_number)

    def get_head_track_names(self):
        """
        Returns list containing the first track name in each track group.
        """
        head_track_names = []
        for group in self.track_groups:
            head_track_names.append(group[0].name)
        return head_track_names

       
    def generate_collection_names(self):
        """
        Generate names of NameCollection objects. 
        """
        COLLECTION_NAME_FORMAT_STRING = "collection_seg_{track_name}"
        head_track_names = self.get_head_track_names()
        names = []
        for name in head_track_names:
            names.append(COLLECTION_NAME_FORMAT_STRING.format(track_name=name))
        return names 

    def generate_name_collection_entries(self):
        """
        Generate entries in NameCollection objects. 
        """
        COLLECTION_ENTRY_FORMAT_STRING = \
            "mx_seg{seg_index}_subseg{subseg_index}_{track_name}"
        head_track_names = self.get_head_track_names()
        names = []
        for name in head_track_names:
            for i in range(self.num_segs):
                for j in range(self.num_subsegs):
                    names.append(COLLECTION_ENTRY_FORMAT_STRING.format(seg_index=i, 
                                                                       subseg_index=j, 
                                                                       track_name=name))
        return names
                     
    def generate_tied_covar_object_names(self):
        """
        Generate tied Covar object names. 
        """      
        TIED_COVAR_FORMAT_STRING = "covar_{track_name}{suffix}"
        head_track_names = self.get_head_track_names()
        names = []
        for component_number in range(self.num_mix_components):
            for track_name in head_track_names:
                component_suffix = self.get_template_component_suffix(component_number)
                names.append(TIED_COVAR_FORMAT_STRING.format(track_name=track_name, 
                                                             suffix=component_suffix))
        return names
      
    def generate_covar_object_names(self):
        """
        Generate tied Covar object names. 
        """      
        COVAR_FORMAT_STRING = "covar_seg{seg_index}_subseg{subseg_index}_{track_name}{suffix}"
        head_track_names = self.get_head_track_names()
        names = []
        for component_number in range(self.num_mix_components):
            for i in range(self.num_segs):
                for j in range(self.num_subsegs):
                    for track_name in head_track_names:
                        component_suffix = self.get_template_component_suffix(component_number)
                        names.append(COVAR_FORMAT_STRING.format(seg_index=i, 
                                                                subseg_index=j, 
                                                                track_name=track_name, 
                                                                suffix=component_suffix))

        return names 
    
    def generate_mean_object_names(self):
        """
        Generate Mean object names. 
        """      
        MEAN_FORMAT_STRING = "mean_seg{seg_index}_subseg{subseg_index}_{track_name}{suffix}"
        head_track_names = self.get_head_track_names()
        names = []
        for component_number in range(self.num_mix_components):
            for i in range(self.num_segs):
                for j in range(self.num_subsegs):
                    for track_name in head_track_names:
                        component_suffix = self.get_template_component_suffix(component_number)
                        names.append(MEAN_FORMAT_STRING.format(seg_index=i, 
                                                               subseg_index=j, 
                                                               track_name=track_name, 
                                                              suffix=component_suffix))

        return names
      
    def generate_mx_object_names(self):
        """
        Generate MX object names. 
        """      
        MX_FORMAT_STRING = "mx_seg{seg_index}_subseg{subseg_index}_{track_name}"
        head_track_names = self.get_head_track_names()
        names = []
        for i in range(self.num_segs):
            for j in range(self.num_subsegs):
                for track_name in head_track_names:
                    names.append(MX_FORMAT_STRING.format(seg_index=i, 
                                                         subseg_index=j, 
                                                         track_name=track_name))

        return names
    
    def generate_diag_gaussian_mc_object_names(self):
        """
        Generate DiagGaussianMC object names. 
        """      
        DIAG_GAUSSIAN_FORMAT_STRING = \
           "mc_{distribution}_seg{seg_index}_subseg{subseg_index}_{track_name}{suffix}"
        head_track_names = self.get_head_track_names()
        names = []
        for component_number in range(self.num_mix_components):
            for i in range(self.num_segs):
                for j in range(self.num_subsegs):
                    for track_name in head_track_names:
                        component_suffix = self.get_template_component_suffix(component_number)
                        names.append(DIAG_GAUSSIAN_FORMAT_STRING.format(distribution=self.distribution, 
                                                                    seg_index=i,
                                                                        subseg_index=j, 
                                                                        track_name=track_name, 
                                                                   suffix=component_suffix))
        return names
    
    def generate_gamma_mc_object_names(self):
        GAMMA_MC_FORMAT_STRING = \
            "mc_gamma_seg{seg_index}_subseg{subseg_index}_{track_name}"
        names = []
        head_track_names = self.get_head_track_names()
        for i in range(self.num_segs):
            for j in range(self.num_subsegs):
                for track_name in head_track_names:
                    names.append(GAMMA_MC_FORMAT_STRING.format(seg_index=i,
                                                               subseg_index=j,
                                                               track_name=track_name))
        return names
      
    def generate_dpmf_object_names(self):
        """
        Generate DPMF object names. 
        """      
        # to do for num_mix_components > 1: 
        names = []
        if self.num_mix_components == 1:
            names.append("dpmf_always")
        else: 
            # TODO (with dirichlet extra rows) 
            names.append("")
          
        return names 
      
    def generate_gmtk_object_names(self, gmtk_object_type):
        """
        Generate GMTK object names for the types:
        name of NameCollection: "collection_names"
        entries in NameCollection: "collection_entries"
        Covar: "covar", "tied_covar"
        Mean: "mean"
        MX: "mx"
        MC: "diag_gaussian_mc"
        DPMF: "dpmf"
        :param gmtk_object_type: str: type of gmtk object for which names must be generated
        :return: list[str]: list of GMTK object names 
        """
        GMTK_OBJECT_NAME_GENERATORS = {'mx': self.generate_mx_object_names, 
                                       'diag_gaussian_mc': self.generate_diag_gaussian_mc_object_names, 
                                      'mean': self.generate_mean_object_names, 
                                      'covar': self.generate_covar_object_names,
                                       'tied_covar': self.generate_tied_covar_object_names,
                                      'collection_names': self.generate_collection_names, 
                                       'collection_entries': self.generate_name_collection_entries, 
                                      'dpmf': self.generate_dpmf_object_names}
            
        return GMTK_OBJECT_NAME_GENERATORS[gmtk_object_type]()

    def generate_name_collection(self):
        """
        Generate string representation of NameCollection objects in input master. 
        """ 
        # generate list of collection names
        # num_track_groups (i.e. one for each head track) number 
        # of collection names generated 
        collection_names = self.generate_gmtk_object_names("collection_names")
        # generate list of all names in NameCollections
        # (num_segs * num_subsegs) number of names generated 
        names = self.generate_gmtk_object_names("collection_entries")
        num_track_groups = self.num_track_groups 
        len_name_group = int(len(names) / num_track_groups)
        # names grouped by collection
        name_groups = [names[i:i + len_name_group] for i in
                       range(0, len(names), len_name_group)]
        # create NameCollection objects and add to input master
        for group_index in range(len(name_groups)):
            input_master.name_collection[collection_names[group_index]] = \
                NameCollection(name_groups[group_index])

        return str(input_master.name_collection)

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

    def generate_mean_objects(self):
        """
        Generate string representation of Mean objects in input master. 
        """
        # generate list of names of Mean objects
        names = self.generate_gmtk_object_names("mean")
        means = self.make_mean_data()  # array
        num_track_groups = self.num_track_groups  # number of head track names 
        # dimensions of means: num_segs x num_subsegs x num_head_tracks
        # create Mean objects
        names_array = array(names).reshape((self.num_segs, self.num_subsegs, len(self.track_groups)))
        for i in range(self.num_segs):
            for j in range(self.num_subsegs):
                for k in range(num_track_groups):
                    input_master.mean[names_array[i, j, k]] = Mean(means[i, j, k])
                    
        return str(input_master.mean)

    def generate_covar_objects(self):
        """
        Generate string representation of Covar objects in input master.
        """
        if COVAR_TIED:
            names = self.generate_gmtk_object_names("tied_covar")
        else:
            names = self.generate_gmtk_object_names("covar")
        covar_values = self.vars  # array of variance values
        # creating Covar objects and adding them to input master
        covar_objects = map(Covar, covar_values) 
        input_master.covar.update(dict(zip(names, covar_objects)))

        return str(input_master.covar)

    def generate_mc_objects(self):
        """
        Generate string representation of MC objects in input master. 
        """
        # if distribution is norm or asinh_norm, TODO for missing, gamma
        if self.distribution in DISTRIBUTIONS_LIKE_NORM:
            if USE_MFSDG:
                # TODO
                option = "missing_mc"
            else:
                option = "diag_gaussian_mc"
            # generate MC object names
            names = self.generate_gmtk_object_names(option)
            
            # replicate covar names for iteration
            covar_names = list(input_master.mc.covar) * (
                        self.num_segs * self.num_subsegs)
            # list of all mean names
            mean_names = list(input_master.mc.mean)

            # create MC objects and add them to input master
            mc_objects = []
            for mean_name, covar_name in zip(mean_names, covar_names):
                mc_objects.append(DiagGaussianMC(mean=mean_name, covar=covar_name))
            input_master.mc.update(dict(zip(names, mc_objects)))
            
            return str(input_master.mc)

    def generate_mx_objects(self):
        """Generate string representation of MX objects in input master. 
        """
        # generate list of MX names
        names = self.generate_gmtk_object_names("mx")
        mc_names = list(input_master.mc)  # list of all mc names
        dpmf_names = list(input_master.dpmf)  # list of all dpmf names
        multiple = int(len(names) / len(dpmf_names))
        dpmf_names *= multiple  # replicate dpmf names for iteration
        mx_objects = []
        # parameters required for creating MX object: names of mc, dpmf
        for mc_name, dpmf_name in zip(mc_names, dpmf_names):
            mx_objects.append(MX(dpmf=dpmf_name, components=mc_name))
            
        # adding MX object to input master     
        input_master.mx.update(dict(zip(names, mx_objects)))
        return input_master.mx.__str__()

    def generate_dpmf_objects(self):
        """Generate string representation of DPMF objects in input master. 
        """
        # generate a list of dpmf names
        names = self.generate_gmtk_object_names("dpmf")
        # if single dpmf
        if self.num_mix_components == 1:
            input_master.dpmf[names[0]] = DPMF(1.0)
        else:
            # uniform probabilities
            dpmf_values = str(round(1.0 / self.num_mix_components,
                                    ROUND_NDIGITS))
            # creating DPMF objects and adding them to input master
            dpmf_objects = map(DPMF, dpmf_values) 
            input_master.dpmf.update(dict(zip(names, dpmf_objects)))
            
        return str(input_master.dpmf)

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
            input_master.dense_cpt[names[i]] = np.squeeze(DenseCPT(prob[i]), axis=0)
        # adding dirichlet row if necessary 
        if self.len_seg_strength > 0:
            dirichlet_row = ["DirichletTable %s" % self.make_dirichlet_name(NAME_SEGCOUNTDOWN_SEG_SEGTRANSITION)]
            input_master.dense_cpt[NAME_SEGCOUNTDOWN_SEG_SEGTRANSITION].extra_rows = dirichlet_row
        return str(input_master.dense_cpt)

    def make_dinucleotide_table_row(self):
        pass

    def make_seg_dinucleotide(self):
        pass

    def make_dirichlet_name(self, name):
        return "dirichlet_{}".format(name)

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

    def generate_dirichlet_objects(self):
        # XXX: these called functions have confusing/duplicative names
        if self.len_seg_strength > 0:
            lines = ["DIRICHLET_TAB_IN_FILE inline"]
            lines.append("1\n")  # only one DirichletTab for segCountDown_seg_segTransition 
            row = ["0"] # index of dirichlet tab 
            row.append(self.make_dirichlet_name(NAME_SEGCOUNTDOWN_SEG_SEGTRANSITION))
            # name of dirichlet tab
            dirichlet_table = self.make_dirichlet_table()
            dim_shape = [dirichlet_table.ndim]
            dim_shape.extend(dirichlet_table.shape)
            row.append(" ".join(map(str, dim_shape)))
            lines.append(" ".join(row))
            value = array2text(dirichlet_table)
            lines.append("{}\n\n".format(value))
            return "\n".join(lines)
        else:
            return ""
          
          
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
        dirichlet_spec = param_spec.generate_dirichlet_objects()
        dense_cpt_spec = param_spec.generate_dense_cpt_objects()

        # seg_seg
        num_free_params += fullnum_subsegs * (fullnum_subsegs - 1)

        # segCountDown_seg_segTransition
        num_free_params += fullnum_subsegs
        name_collection_spec = param_spec.generate_name_collection()

        distribution = self.distribution
        if distribution in DISTRIBUTIONS_LIKE_NORM:
            mean_spec = param_spec.generate_mean_objects()
            covar_spec = param_spec.generate_covar_objects()

            # TODO: class RealMatParamSpec 
            # for now this is sufficient because util.USE_MFSDG = False by default 
            real_mat_spec = ""
            mc_spec = param_spec.generate_mc_objects()

            if COVAR_TIED:
                num_free_params += (fullnum_subsegs + 1) * num_track_groups
            else:
                num_free_params += (fullnum_subsegs * 2) * num_track_groups
                
        # TODO: gamma distribution option 

        else:
            raise ValueError("distribution %s not supported" % distribution)

        dpmf_spec = param_spec.generate_dpmf_objects()
        mx_spec = param_spec.generate_mx_objects()
        card_seg = num_segs
        ve_spec = VirtualEvidenceSpec(self)

        return locals()  # dict of vars set in this function