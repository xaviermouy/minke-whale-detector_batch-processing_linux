#!/bin/bash

# Notes:
# When editing this file make sure it uses the Unix end-of-line 
# In Notepad++: Edit > EOL Conversion > Unix (LF)

set -e
SCRIPT_DIR3=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo $SCRIPT_DIR3
cd /home/nefsc/Documents
python3.9 -m venv ketos-env
source ketos-env/bin/activate
pip install --upgrade pip --no-cache-dir
pip install -r $SCRIPT_DIR3/requirements.txt --no-cache-dir