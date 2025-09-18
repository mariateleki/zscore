import unittest
from pathlib import Path

from zscore import utils_dirs

class TestUtilsDirs(unittest.TestCase):
    def test_base_dir_is_two_parents_up(self):
        expected = Path(utils_dirs.__file__).resolve().parents[2]
        self.assertEqual(utils_dirs.BASE_DIR, expected)

    def test_data_dir_is_child_of_base(self):
        self.assertEqual(utils_dirs.DATA_DIR, utils_dirs.BASE_DIR / "data")

    def test_treebank_dir_structure(self):
        self.assertEqual(
            utils_dirs.TREEBANK_DIR,
            utils_dirs.DATA_DIR / "treebank_3"
        )
        self.assertEqual(
            utils_dirs.TREEBANK_MRG_DIR,
            utils_dirs.TREEBANK_DIR / "parsed" / "mrg" / "swbd"
        )

    def test_sample_mrg_path(self):
        expected = utils_dirs.TREEBANK_MRG_DIR / "4" / "sw4004.mrg"
        self.assertEqual(utils_dirs.TREEBANK_SAMPLE_MRG_FILE, expected)

    def test_all_are_paths(self):
        self.assertIsInstance(utils_dirs.BASE_DIR, Path)
        self.assertIsInstance(utils_dirs.DATA_DIR, Path)
        self.assertIsInstance(utils_dirs.TREEBANK_DIR, Path)
        self.assertIsInstance(utils_dirs.TREEBANK_MRG_DIR, Path)
        self.assertIsInstance(utils_dirs.TREEBANK_SAMPLE_MRG_FILE, Path)


if __name__ == "__main__":
    unittest.main()
