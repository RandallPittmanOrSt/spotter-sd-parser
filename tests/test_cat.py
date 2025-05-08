#!/usr/bin/env python3 -m unittest -v

"""
test the concatenation function of parser
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from spotter_sd_parser.sd_file_parser import cat


class CatTest(unittest.TestCase):
    def testCatSst(self):
        for suffix in self.suffixes:
            # just SST right now
            output_file_path = Path(self.outputpath, f"{self.outFiles[suffix]}.csv")
            result = cat(
                path=self.inputpath, output_file_path=output_file_path, Suffix=suffix
            )

    def setUp(self):
        """
        prepare for running the parser
        """
        # note relative path
        self.inputpath = Path("example_data/2021-01-15")
        self.inputfn = "0235_SST.CSV"
        self.inputfqfn = os.path.join(self.inputpath, self.inputfn)

        self.outputpath = tempfile.mkdtemp()

        # borrowed from parser script main()
        # self.suffixes = ['FLT','SPC','SYS','LOC','GPS','SST']
        self.suffixes = ["SST"]
        self.outFiles = {
            "FLT": "displacement",
            "SPC": "spectra",
            "SYS": "system",
            "LOC": "location",
            "GPS": "gps",
            "SST": "sst",
        }
        if not self.outputpath:
            raise ValueError("problem creating temporary output directory")

    def tearDown(self):
        """
        delete temporary output file(s)
        """
        if os.path.exists(self.outputpath):
            print(f"cleanup: deleting {self.outputpath}")
            shutil.rmtree(self.outputpath)


if __name__ == "__main__":
    unittest.main()
