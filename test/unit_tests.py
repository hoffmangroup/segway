import unittest

from segway.observations import merge_windows


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


if __name__ == "__main__":
    unittest.main()
