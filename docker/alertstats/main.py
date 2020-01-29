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
    dest = pathlib.Path(args.output_dir, args.output_file)
    with open(dest, "w") as output:
        for f in args.files:
            status = 0
            err = None
            try:
                track = np.load(f)
                start = time.time()
                result = alert([track])
            except Exception as e:
                status = 1
                err = e
            finally:
                end = time.time()
                delta = end - start
            result = [str(delta), str(status), str(len(track)), err.__class__.__name__]
            print(result)
            output.write(",".join(result))
            output.write("\n")
            output.flush()