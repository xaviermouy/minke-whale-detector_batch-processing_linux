#!/bin/bash

# Notes:
# When editing this file make sure it uses the Unix end-of-line 
# In Notepad++: Edit > EOL Conversion > Unix (LF)

set -e

cd /home/nefsc/Documents
python3.9 -m venv ketos-env
source ketos-env/bin/activate
pip install --upgrade pip
pip install numpy==1.26.4
pip install soundfile>=0.10
pip install ketos==2.6.2
pip install ecosound==0.0.25
pip install pandas==2.0.2
