from cyclopts import App

from spotter_sd_parser import sd_file_parser, smartmooring_pd

app = App(name="spotter-sd-parser", help_on_error=True)
app.default(sd_file_parser.main)
app.command(smartmooring_pd.main, name="smartmooring-pd")
