import configparser
import os.path

import pfhedge


def test_version():
    parser = configparser.ConfigParser()
    parser.read(os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"))
    assert pfhedge.__version__ == parser["tool.poetry"]["version"].replace('"', "")
