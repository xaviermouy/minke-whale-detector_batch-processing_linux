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

# EDIT THIS SECTION FOR EACH DEPLOYMENT
AUDIO_DIR=/mnt/PassiveAcoustics/Stellwagen_Old/DETECTORS/MinkePulseTrains_Ketos/v0.2/demo/data/ERROR/deployment3
FILES_EXT=.aif
CHANNEL=6
RECORDING_ID=USA-NEFSC-GA-201612-CH6
OUTPUT_DIR=/mnt/PassiveAcoustics/Stellwagen_Old/DETECTORS/MinkePulseTrains_Ketos/v0.2/demo/results/deployment3
CREATE_DAILY_SUMMARY=true # true or false
SUMMARY_TIME_OFFSET=0 # in hours (can be positive of negative)

# DO NOT EDIT THIS SECTION
/mnt/PassiveAcoustics/Stellwagen_Old/DETECTORS/MinkePulseTrains_Ketos/v0.2/scripts/run_ketos_detector.sh

