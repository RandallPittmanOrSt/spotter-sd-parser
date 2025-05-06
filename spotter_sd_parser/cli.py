from cyclopts import App

from spotter_sd_parser import sd_file_parser

app = App()
app.default(sd_file_parser.main)
