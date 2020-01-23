from __future__ import absolute_import
from os import remove
from os.path import isfile, sep
from tempfile import gettempdir
import unittest

from numpy import allclose, array, empty

from segway.observations import (merge_windows, VirtualEvidenceWarning,
                                 get_downsampled_virtual_evidence_data_and_presence)
from segway.task import prepare_gmtk_observations, prepare_virtual_evidence
from segway._util import EXT_INT, EXT_FLOAT, EXT_VIRTUAL_EVIDENCE


class TestObservations(unittest.TestCase):

    def test_merge_windows(self):
        self.assertEqual(
            merge_windows([(0, 1)]),
            [(0, 1)])

        self.assertEqual(
            merge_windows([(0, 100), (100, 200)]),
            [(0, 200)])

        self.assertEqual(
            merge_windows([(0, 100), (101, 200)]),
            [(0, 100), (101, 200)])

        self.assertEqual(
            merge_windows([(0, 100), (50, 125), (99, 150), (200, 201)]),
            [(0, 150), (200, 201)])


class TestTask(unittest.TestCase):

    def test_prepare_observations(self):
        gmtk_args = ["baz", "-of1", "foo", "-of2", "bar", "-cppCommandOptions","-DVIRTUAL_EVIDENCE_LIST_FILENAME=VE_PLACEHOLDER"]

        [float_obs_filename, int_obs_filename, virtual_evidence_filename,
         float_obs_list_filename, int_obs_list_filename, virtual_evidence_list_filename] = \
            prepare_gmtk_observations(gmtk_args, "chr1", 0, 8000, empty(0), 1,
                                      None)

        current_tmp_dir = gettempdir()
        # Test proper naming of the observation files
        proper_obs_filename_prefix = current_tmp_dir + sep + "chr1.0.8000"
        self.assertTrue(float_obs_filename.startswith(
                            proper_obs_filename_prefix))
        self.assertTrue(float_obs_filename.endswith(EXT_FLOAT))

        self.assertTrue(int_obs_filename.startswith(
                            proper_obs_filename_prefix))
        self.assertTrue(int_obs_filename.endswith(EXT_INT))

        self.assertTrue(virtual_evidence_filename.startswith(
                            proper_obs_filename_prefix))
        self.assertTrue(virtual_evidence_filename.endswith(EXT_VIRTUAL_EVIDENCE))

        # Test filenames exist in the given environment's tempdir
        self.assertTrue(float_obs_list_filename.startswith(current_tmp_dir))
        self.assertTrue(int_obs_list_filename.startswith(current_tmp_dir))
        self.assertTrue(virtual_evidence_list_filename.startswith(current_tmp_dir))
        self.assertTrue(float_obs_filename.startswith(current_tmp_dir))
        self.assertTrue(int_obs_filename.startswith(current_tmp_dir))
        self.assertTrue(virtual_evidence_filename.startswith(current_tmp_dir))

        # Check if the observation files exist
        self.assertTrue(isfile(float_obs_filename))
        self.assertTrue(isfile(int_obs_filename))
        self.assertTrue(isfile(virtual_evidence_filename))

        # Check the first line in the list contains their respective
        # observation file (and type)
        with open(float_obs_list_filename) as float_list_file:
            line = float_list_file.readlines()[0].strip()
            self.assertTrue(line.startswith(proper_obs_filename_prefix))
            self.assertTrue(line.endswith(EXT_FLOAT))

        with open(int_obs_list_filename) as int_list_file:
            line = int_list_file.readlines()[0].strip()
            self.assertTrue(line.startswith(proper_obs_filename_prefix))
            self.assertTrue(line.endswith(EXT_INT))

        with open(virtual_evidence_list_filename) as virtual_evidence_list_file:
            line = virtual_evidence_list_file.readlines()[0].strip()
            self.assertTrue(line.startswith(proper_obs_filename_prefix))
            self.assertTrue(line.endswith(EXT_VIRTUAL_EVIDENCE))
        # Check gmtk args were modified for the temp list files
        # -of1 option
        self.assertEqual(gmtk_args[2], float_obs_list_filename)
        # -of2 option
        self.assertEqual(gmtk_args[4], int_obs_list_filename)
        # virtual evidence cpp directive
        self.assertEqual(gmtk_args[6], "-DVIRTUAL_EVIDENCE_LIST_FILENAME=%s" % virtual_evidence_list_filename)

        remove(float_obs_list_filename)
        remove(int_obs_list_filename)
        remove(virtual_evidence_list_filename)
        remove(int_obs_filename)
        remove(float_obs_filename)
        remove(virtual_evidence_filename)


class TestVirtualEvidence(unittest.TestCase):
    def test_proper_downsampling(self):
        virtual_evidence_coords = "[(0, 4), (0, 2), (2, 4)]"
        virtual_evidence_priors = "[{0: 0.1}, {1: 0.5}, {2: 0.7}]"

        raw = prepare_virtual_evidence(1, 0, 5, 4,
                                       virtual_evidence_coords,
                                       virtual_evidence_priors)
        expected = array([[0.1, 0.5, 0.2, 0.2], [0.1, 0.5, 0.2, 0.2],
                          [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.7, 0.1],
                          [0, 0, 0, 0]])
        self.assertTrue(allclose(raw, expected))

        downsampled, presence = get_downsampled_virtual_evidence_data_and_presence(
            raw, resolution=5, num_labels=4)
        self.assertTrue(allclose(downsampled, array([0.1, 0.3, 0.45, 0.15])))
        print(presence)

    def test_overlapping_labels(self):
        virtual_evidence_priors = "[{0: 0.2}, {0: 0.7}]"
        virtual_evidence_coords = "[(0, 100), (50, 150)]"

        with self.assertRaises(ValueError):
            prepare_virtual_evidence(1, 0, 800, 4,
                                     virtual_evidence_coords,
                                     virtual_evidence_priors)

    def test_zero_labels(self):
        virtual_evidence_coords = "[(0, 4), (4, 5)]"
        virtual_evidence_priors = "[{1: 0.8}, {0: 0}]"

        expected = prepare_virtual_evidence(1, 0, 5, 3,
                                           virtual_evidence_coords,
                                           virtual_evidence_priors)
        self.assertTrue((expected[4] == array([0, 0.5, 0.5])).all())

    def test_negative_priors(self):
        virtual_evidence_coords = "[(0, 4), (2, 5)]"
        virtual_evidence_priors = "[{1: -0.8}, {0: 0.7}]"

        with self.assertRaises(ValueError):
            prepare_virtual_evidence(1, 0, 10, 4,
                                     virtual_evidence_coords,
                                     virtual_evidence_priors)

    def test_priors_over_one(self):
        virtual_evidence_coords = "[(0, 4), (2, 5)]"
        virtual_evidence_priors = "[{1: 0.8}, {0: 0.7}]"

        # Only python 3 supports assetWarns, passes if there is an
        # AttributeError in python 2
        # Should be dropped once python 2 support is
        try:
            with self.assertWarns(VirtualEvidenceWarning):
                prepare_virtual_evidence(1, 0, 10, 4,
                                         virtual_evidence_coords,
                                         virtual_evidence_priors)
        except AttributeError:
            pass


if __name__ == "__main__":
    unittest.main()
