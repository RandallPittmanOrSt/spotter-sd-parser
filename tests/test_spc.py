#!/usr/bin/env python3 -m unittest

"""
test spectral file parsing
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from spotter_sd_parser.parsing import parseSpectralFiles


class SpectralParsingTest(unittest.TestCase):
    def testExplicitOutputParserRun(self):
        """
        run the parser on example data, while customizing which
        output files we want
        """
        parseSpectralFiles(
            input_file_path=self.inputfn,
            output_dir=self.outputpath,
            outputSpectra=["Szz", "Sxx"],
        )
        # did the output files get created?
        # only Szz gets created by default (see lines 827-829)
        self.assertTrue(os.path.exists(os.path.join(self.outputpath, "Szz.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.outputpath, "Sxx.csv")))

    def testBasicParserRun(self):
        """
        run the parser on example data, using default parameters
        """
        parseSpectralFiles(input_file_path=self.inputfn, output_dir=self.outputpath)
        # did the output file get created?
        # only Szz gets created by default (see lines 827-829)
        self.assertTrue(os.path.exists(os.path.join(self.outputpath, "Szz.csv")))

    def setUp(self):
        """
        prepare for running the parser
        """
        self.inputfn = Path("example_data/2021-01-15/0235_SPC.CSV")
        # self.outputfn = f"spctest_{ceil(time.time()):x}.csv"
        self.outputpath = Path(tempfile.mkdtemp())
        if not self.outputpath:
            raise ValueError("problem creating temporary output directory")

    def tearDown(self):
        """
        delete temporary output file
        """
        # if os.path.exists(self.outputfn):
        #     os.remove(self.outputfn)
        if os.path.exists(self.outputpath):
            print(f"cleanup: deleting {self.outputpath}")
            shutil.rmtree(self.outputpath)


if __name__ == "__main__":
    unittest.main()
