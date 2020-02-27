from __future__ import print_function

import argparse
import time
import pathlib
import traceback


from online_detection import alert

import numpy as np
import sesamelib.utils as cli_utils

OUTPUT_DIR = "/tmp/result"
OUTPUT_FILE = "stats.csv"

if __name__ == "__main__":
    def create_result_dir(dir):
        dir = pathlib.Path(args.output_dir, dir)
        dir.mkdir(parents=True, exist_ok=True)
        return dir

    parser = argparse.ArgumentParser(description="Benchmark alert function")
    parser.add_argument('files', metavar='TRACK', type=str, nargs='+',
                    help='list of tacks (numpy files) to test')
    parser.add_argument('--output_dir', '-o', type=str,
                        help="Output dir",
                        default=cli_utils.lookup_env("OUTPUT_DIR", OUTPUT_DIR))
    parser.add_argument('--output_file', '-f', type=str,
                        help="Output file",
                        default=cli_utils.lookup_env("OUTPUT_FILE", OUTPUT_FILE))
    args = parser.parse_args()
    print(args)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args = parser.parse_args()
    track = []
    stats = pathlib.Path(args.output_dir, args.output_file)
    normal_dir = create_result_dir("normal")
    abnormal_dir = create_result_dir("abnormal")
    faulty_dir = create_result_dir("faulty")
    keys = ["track",
            "normality",
            "duration",
            "status",
            "length",
            "start",
            "end",
            "err",
            "file"]
    with open(stats, "w") as output:
        output.write(",".join(keys))
        output.write("\n")
        for f in args.files:
            status = 0
            err = None
            track_name = pathlib.Path(f).name.split(".")[0]
            result = ""
            start = time.time()
            track = np.load(f)
            try:
                t = alert([track])
                result = "normal"
                track_dir = normal_dir
                if len(t) > 0:
                    result = "abnormal"
                    track_dir = abnormal_dir
            except Exception as e:
                status = 1
                err = e
                track_dir = faulty_dir
            finally:
                end = time.time()
                delta = end - start
                track_file = pathlib.Path(track_dir, f"{track_name}.npy")
                np.save(track_file, track)
            # file format is something like
            # <start>-<end>-<mmsi>-<track_index>
            splitted_name = track_name.split("-")
            start = splitted_name[0]
            end = splitted_name[1]
            result = [
                track_name,
                result,
                str(delta),
                str(status),
                str(len(track)),
                str(start),
                str(end),
                err.__class__.__name__,
                f
            ]
            print(result)
            output.write(",".join(result))
            output.write("\n")
            output.flush()