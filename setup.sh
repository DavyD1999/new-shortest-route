#!/bin/bash
set -x # shows what is being executed
rm -rf env
#don't forget chmod +x

rm -rf /opt/virtualenvs/python3/lib/python3.8/site-packages/typing.py # only needed temporarily to fix bug in pip and typing pip uninstall typing does not work
pip uninstall pip --y # dash dash y tells yes to everything
python -m pip install --upgrade pip --user
python3.8 -m virtualenv env # default is python 3.8.10
source env/bin/activate
pip3.8 install -r requirements.txt