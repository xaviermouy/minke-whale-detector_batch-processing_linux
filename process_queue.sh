#!/bin/bash

# Notes:
# When editing this file make sure it uses the Unix end-of-line 
# In Notepad++: Edit > EOL Conversion > Unix (LF)

# Input arguments
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
QUEUE_DIR="$SCRIPT_DIR/processing_queue"
PROCESSED_DIR="$SCRIPT_DIR/processing_queue/processed"
ERROR_DIR="$SCRIPT_DIR/processing_queue/errors"

# Check that processed ansd errors folders exist
[ -d $PROCESSED_DIR ] || mkdir -p $PROCESSED_DIR
[ -d $ERROR_DIR ] || mkdir -p $ERROR_DIR

# Starts processing queue
COUNT=0
for JOB in $QUEUE_DIR/*.sh; do
	(( COUNT++ ))
	echo ' '
	echo '================================================== '
    echo '               Processing job ' $COUNT '                 '
	echo '================================================== '
	echo ' '
	echo $JOB
	$JOB
	SUCCESS=$?
	if [ $SUCCESS -eq 0 ]; then
		mv -f $JOB $PROCESSED_DIR
	else
		echo "Job failed..."
		mv $JOB $ERROR_DIR
	fi
done

echo ' '
echo '----------- All jobs have completed --------------'
echo ' '

read -p "Press any key to close window ..."