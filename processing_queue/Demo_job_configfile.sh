#!/bin/bash

# #####################################################################
#                         Processing Job Script
#                         Minke whale detector
# #####################################################################

# Description:
#	This file defines all the input arguments for running the minke whale
#   detector. One of these files defines one processing job. Create one 
#   of these files for each deployment to process and place them in the
#   "processing_queue" folder. To start processing the jobs, execute the
#   script process_queue.sh in the minke whael detector container.
#	
# Note 1:
#   When editing this file make sure it uses the Unix end-of-line 
#   In Notepad++: Edit > EOL Conversion > Unix (LF)
#
# Note 2:
#   All paths should use the forward slash (/)
#   To point to Stellwagen use /mnt/stellwagen/
#
# Note 3:
#   There should not be any spaces on either side of the equal sign when 
#   defining input argunments. For example:
#      AUDIO_DIR=/my/audio/path is correct
#      AUDIO_DIR = /my/audio/path is NOT correct
#
# Note 4:
#   "set -a" should always be at the top of the script
#
# Note 5:
#   The last section of the file (# Starts detector) should not be edited 
#   unless a different version of the detctor needs to be used.
#
# #####################################################################

set -a

# Input arguments
AUDIO_DIR=/mnt/PassiveAcoustics_Soundfiles/BOTTOM_MOUNTED/NAVFAC_NC/NAVFAC_NC_201806_HAT04B/HAT_B_04_01_2kHz_Sampled
FILES_EXT=.aif
CHANNEL=6
RECORDING_ID=demo_deployment
OUTPUT_DIR=/mnt/PassiveAcoustics/DETECTOR_OUTPUT/PYTHON_MINKE/KETOS_v0.2/Raw/NAVFAC_NC/NAVFAC_NC_201806_HAT04B
CREATE_DAILY_SUMMARY=true # true or false
SUMMARY_TIME_OFFSET=-5 # in hours (can be positive of negative)

# Starts detector
/mnt/PassiveAcoustics/DETECTORS_SOFTWARE/PYTHON_MINKE/v0.2/scripts/run_ketos_detector.sh

