#!/usr/bin/env python3 -m unittest

"""
test location file parsing
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from spotter_sd_parser.parsing import parseLocationFiles


class LocationParsingTest(unittest.TestCase):
    def testBasicParserRun(self):
        parseLocationFiles(
            input_file_path=self.inputfn, kind="LOC",
            output_file_path=self.output_dir / self.outputfn
        )
        self.assertTrue(os.path.exists(self.output_dir / self.outputfn))
        self.assertFalse(os.path.exists("displacement.csv"))
        self.assertFalse(os.path.exists(self.output_dir / "displacement.csv"))

    def setUp(self):
        """
        prepare for running the parser
        """
        self.inputfn = Path("example_data/2021-01-15/0235_LOC.CSV")
        # self.outputfn = f"spctest_{ceil(time.time()):x}.csv"
        self.outputfn = "location.csv"
        self.output_dir = Path(tempfile.mkdtemp())
        if not self.output_dir:
            raise ValueError("problem creating temporary output directory")

    def tearDown(self):
        """
        delete temporary output file(s)
        """
        if os.path.exists(self.output_dir):
            print(f"cleanup: deleting {self.output_dir}")
            shutil.rmtree(self.output_dir)


if __name__ == "__main__":
    unittest.main()
