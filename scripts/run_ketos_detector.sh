#!/bin/bash

# Notes:
# When editing this file make sure it uses the Unix end-of-line 
# In Notepad++: Edit > EOL Conversion > Unix (LF)

set -e


## Edit with model and confidence threshold ##############################
MODEL=/mnt/PassiveAcoustics/DETECTORS_SOFTWARE/PYTHON_MINKE/v0.2/models/spectro-5s_fft-0.256_step-0.03_fmin-0_fmax-800_no-norm/ketos_model.kt
MIN_CONFIDENCE=0.6
##########################################################################

SCRIPT_DIR2=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo $SCRIPT_DIR2

# check is python environment is already setup. If not, creates it
echo 'Verification of the python environment'
if [ -d "/home/nefsc/Documents/ketos-env" ]; then
echo "Python environment already setup."
else
echo "Python environment not configured. Creating new python environment."
"$SCRIPT_DIR2/create_python_environment.sh"
fi
# || echo "Python environment not configured. Creating new python environment." && "$SCRIPT_DIR2/create_python_environment.sh"
echo 'Activating environment'
source /home/nefsc/Documents/ketos-env/bin/activate
echo 'Starting minke whale detector on' $AUDIO_DIR
python "$SCRIPT_DIR2/run_ketos_detector.py" --audio_folder=$AUDIO_DIR --extension=$FILES_EXT --channel=$CHANNEL --deployment_id=$RECORDING_ID --output_folder=$OUTPUT_DIR --model=$MODEL --tmp_dir=/tmp/ --batch_size=512 --threshold=0.1 --step_sec=1 --smooth_sec=0 --recursive 
if [ "$CREATE_DAILY_SUMMARY" = true ] ; then
	echo 'Creating spectrogram and daily summaries'
	python "$SCRIPT_DIR2/create_detection_spectrograms_and_spreadsheet.py" --detec_dir=$OUTPUT_DIR --time_offset=$SUMMARY_TIME_OFFSET --min_confidence=$MIN_CONFIDENCE
fi
echo 'Closing environment'
deactivate
