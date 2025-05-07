#!/usr/bin/env python3 -m unittest

"""
test the SST parsing functions of the sd_file_parser.py
"""

import os
import time
import unittest
from pathlib import Path

from spotter_sd_parser.parsing import parseLocationFiles


class SSTParsingTest(unittest.TestCase):
    def testOutputNameParserRun(self):
        """
        set calling parseLocationFiles() in SST mode
        specify the name of the output file.
        """
        parseLocationFiles(
            input_file_path=self.inputfn, kind="SST", output_file_path=self.outputfn
        )
        self.assertTrue(os.path.exists(self.outputfn))

    def testNoOutputNameParserRun(self):
        """
        set calling parseLocationFiles() in SST mode
        don't give an output name and it will use displacement.csv
        """
        parseLocationFiles(input_file_path=self.inputfn, kind="SST")
        self.assertFalse(os.path.exists(self.outputfn))
        # surprise!@
        self.assertTrue(os.path.exists("displacement.csv"))

    def setUp(self):
        """
        prepare for running the parser
        """
        self.inputfn = Path("example_data/2021-01-15/0235_SST.CSV")
        self.outputfn = Path("sst.csv")

    def tearDown(self):
        """
        delete temporary output file(s)
        """
        for fn in ["displacement.csv", self.outputfn]:
            if os.path.exists(fn):
                print(f"removing {fn}")
                os.remove(fn)


if __name__ == "__main__":
    unittest.main()
