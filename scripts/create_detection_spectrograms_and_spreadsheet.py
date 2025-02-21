# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 08:22:48 2022

@author: xavier.mouy
"""

from ecosound.core.measurement import Measurement
from ecosound.core.annotation import Annotation
import datetime
import os
import argparse
import sqlite3
import pandas as pd
from ecosound.core.tools import filename_to_datetime, list_files

def set_args_parser():
    parser = argparse.ArgumentParser(
        description="Script to extract spectrogranm of detections amd produce daily detection spreadsheet."
    )
    # define command line arguments
    parser.add_argument(
        "--detec_dir",
        type=str,
        default=None,
        help="Path to the detection results",
    )
    parser.add_argument(
        "--time_offset",
        type=int,
        default=0,
        help="Number of hours to add of subtract to timestamp of the file names. Default is 0.",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="minimum confidence value of detections to extract. Default is 0.5",
    )
    return parser

def run():
    # parse command line args
    parser = set_args_parser()
    args = parser.parse_args()

    # in_dir = r"Z:\Stellwagen_Old\STAFF\Xavier\ketos_minke_detector\NEFSC_MA-RI_COX01\NEFSC_MA-RI_202107_COX01"
    in_dir = args.detec_dir
    #sqlite_file = "detections.sqlite"
    aggregate_time_offset = args.time_offset

    # Load dataset
    print("Loading detections...")
    #dataset = Measurement()
    #dataset.from_sqlite(os.path.join(in_dir, sqlite_file))
    dataset = Annotation()
    dataset.from_netcdf(in_dir)

    # load files processed
    #conn = sqlite3.connect(os.path.join(in_dir, sqlite_file))
    #files_list = pd.read_sql_query("SELECT * FROM " + "files_processed", conn)
    #conn.close()
    files_list = list_files(in_dir, suffix='.nc')

    #dates = filename_to_datetime(list(files_list['File_processed']))
    dates = filename_to_datetime(files_list)
    min_date = str(min(dates))
    max_date = str(max(dates))

    # Filter
    print("Filtering detections...")
    dataset.filter("label_class=='MW'", inplace=True)
    dataset.filter("confidence>="+ str(args.min_confidence) , inplace=True)

    # Apply time offset
    dataset.data["time_min_date"] = dataset.data[
        "time_min_date"
    ] + datetime.timedelta(hours=aggregate_time_offset)

    # Create spectrograms and wav files
    print("Extracting detection spectrograms...")
    out_dir = os.path.join(in_dir, "extracted_detections")
    if os.path.isdir(out_dir) == False:
        os.mkdir(out_dir)

    dataset.export_spectrograms(
        out_dir,
        time_buffer_sec=5,
        spectro_unit="sec",
        spetro_nfft=0.256,
        spetro_frame=0.128,
        spetro_inc=0.032,
        freq_min_hz=0,
        freq_max_hz=1000,
        sanpling_rate_hz=2000,
        filter_order=8,
        filter_type="iir",
        fig_size=(15, 10),
        deployment_subfolders=False,
        date_subfolders=True,
        file_name_field="time_max_date",
        # file_name_field="audio_file_name",
        file_prefix_field="confidence",
        channel=None,
        colormap="Greys",  # "viridis",
        save_wav=True,
    )

    # Create daily agggregate.
    print("Creating detections daily aggregates...")
    daily_counts = dataset.calc_time_aggregate_1D(
        integration_time="1D",
        resampler="count",
        start_date= str(min(dates)),
        end_date= str(max(dates)),
    )
    daily_counts.rename(columns={"value": "Detections"}, inplace=True)


    daily_counts.to_csv(
        os.path.join(out_dir, "daily_counts.csv"),
        index_label="Date (UTC" + str(aggregate_time_offset) + ")",
    )

    print("Done!")

if __name__ == '__main__':
    run()