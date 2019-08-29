from __future__ import absolute_import
from os import remove
from os.path import isfile, sep
from tempfile import gettempdir
import unittest

from numpy import array, ndarray, empty
from path import Path

from segway.observations import merge_windows
from segway.task import prepare_gmtk_observations
from segway._util import EXT_INT, EXT_FLOAT

from segway.gmtk.input_master import (Covar, DenseCPT, DeterministicCPT,
                                      InlineSection,
                                      InputMaster, Mean, NameCollection, 
                                      Object)


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
        gmtk_args = ["baz", "-of1", "foo", "-of2", "bar"]

        [float_obs_filename, int_obs_filename,
         float_obs_list_filename, int_obs_list_filename] = \
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

        # Test filenames exist in the given environment's tempdir
        self.assertTrue(float_obs_list_filename.startswith(current_tmp_dir))
        self.assertTrue(int_obs_list_filename.startswith(current_tmp_dir))
        self.assertTrue(float_obs_filename.startswith(current_tmp_dir))
        self.assertTrue(int_obs_filename.startswith(current_tmp_dir))

        # Check if the observation files exist
        self.assertTrue(isfile(float_obs_filename))
        self.assertTrue(isfile(int_obs_filename))

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

        # Check gmtk args were modified for the temp list files
        # -of1 option
        self.assertEqual(gmtk_args[2], float_obs_list_filename)
        # -of2 option
        self.assertEqual(gmtk_args[4], int_obs_list_filename)

        remove(float_obs_list_filename)
        remove(int_obs_list_filename)
        remove(int_obs_filename)
        remove(float_obs_filename)


class TestGMTK(unittest.TestCase):
    def test_input_master(self):
        input_master = InputMaster(InlineSection(DenseCPT("example_cpt", [[0.5, 0.5]])), InlineSection(Object("example_generic_object", "basic decision string", "DT")))
        input_master["DENSE_CPT"].update(DenseCPT("two_dimensional_cpt", [[0.25, 0.25], [0.25, 0.25]]))
        input_master.update(InlineSection(Object("example_generic_object2", "basic decision string", "DT")))
        input_master.update(InlineSection(Mean("mean_duplication", [1.0]),
                                          Mean("mean_no_CNV", [0]),
                                          Mean("mean_deletion", [-1.0])))

        input_master.update(InlineSection(Covar("logR", [0.04])))

        input_master.update(InlineSection(NameCollection("collection_CNV", ["mx_deletion", "mx_normal", "mx_duplication"])))

        input_master.update(InlineSection(DeterministicCPT("frameIndex_ruler", [1, 2000000, 2, "map_seg_segCountDown"])))

        test_master = Path("sample_input.master")
        with open(test_master) as sample_master:
            expected = sample_master.read().rstrip()
        self.assertEqual(str(input_master), expected)


if __name__ == "__main__":
    unittest.main()
