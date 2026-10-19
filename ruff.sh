#!/usr/bin/env bash
. .venv/bin/activate
ruff format sd_file_parser.py tests legacy
ruff check --select I --fix sd_file_parser.py tests legacy
