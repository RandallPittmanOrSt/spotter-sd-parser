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
            input_file_path=self.inputfn, kind="LOC", output_file_path=self.outputfn
        )
        self.assertTrue(os.path.exists(self.outputfn))
        self.assertFalse(os.path.exists("displacement.csv"))

    def setUp(self):
        """
        prepare for running the parser
        """
        self.inputfn = Path("example_data/2021-01-15/0235_LOC.CSV")
        # self.outputfn = f"spctest_{ceil(time.time()):x}.csv"
        self.outputfn = Path("location.csv")
        self.outputpath = Path(tempfile.mkdtemp())
        if not self.outputpath:
            raise ValueError("problem creating temporary output directory")

    def tearDown(self):
        """
        delete temporary output file
        """
        if os.path.exists(self.outputfn):
            print(f"removing {self.outputfn}")
            os.remove(self.outputfn)
        if os.path.exists(self.outputpath):
            print(f"cleanup: deleting {self.outputpath}")
            shutil.rmtree(self.outputpath)


if __name__ == "__main__":
    unittest.main()
