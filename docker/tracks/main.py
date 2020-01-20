from __future__ import print_function

import argparse
import doctest
from functools import partial
import io
from itertools import groupby
import json
import math
from operator import itemgetter, attrgetter
import os
import pathlib
import traceback

import numpy as np
from pyspark.sql import SparkSession
import time

from flags_config_SESAME import config, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, SPEED_MAX
MIN_TIMESPAN = 4 * 60 * 60

from sesamelib.multitask import Track, MaxIntervalError
import sesamelib.utils as cli_utils
from sesamelib.sesame_faust import BaseDynamicMessage

OUTPUT_DIR = "/tmp/trajectories"

# ais types
GLOBAL = "global"
BRITTANY = "brittany"

KEYS=["mmsi", "sog","cog", "x", "y", "tagblock_timestamp", "true_heading"]


def decode(ais_type, e):
    """Decode the message according the the ais_type."""
    def _decode(raw):
        import ais.stream
        f = io.StringIO(raw)
        msg = None
        for msg in ais.stream.decode(f):
            pass
        f.close()
        return msg
    msg = {}
    try:
        if ais_type == GLOBAL:
            msg = _decode(e)
        elif ais_type == BRITTANY:
            day, hour, _e = e.split(" ")
            msg = _decode(_e)
            if msg:
                tagblock_timestamp = time.mktime(time.strptime("%s %s" % (day, hour), "%Y/%m/%d %H:%M:%S"))
                msg.update({"tagblock_timestamp": tagblock_timestamp})
        else:
            raise Exception("ais_type not recognized")
    except:
        print(e)
        traceback.print_exc()
        with open("decoding_errors.txt", "w") as f:
            f.write(e)
    return msg


def extractKeys(msg):
    """Extract mandatory keys. Run after filter1."""
    r = {}
    for key in KEYS:
        r.update({key: msg.get(key)})
    return r

def buildTracks(lat_min_max, lon_min_max, min_timespan, interval_max, max_speed, mmsi_msgs):
    def new_traj():
        return Track(
                    min_timespan=min_timespan,
                    lat_min_max=lat_min_max,
                    lon_min_max=lon_min_max,
                    max_speed=max_speed,
                    max_interval=interval_max
                )
    mmsi, msgs = mmsi_msgs
    split_tracks = []
    track = new_traj()
    split_tracks.append(("%s-%s" % (mmsi, len(split_tracks)), track))
    for msg in msgs:
        try:
            track.add(BaseDynamicMessage(**msg))
        except MaxIntervalError:
            print(f"Creating a new track for {mmsi}")
            track = new_traj()
            split_tracks.append(("%s-%s" % (mmsi, len(split_tracks)), track))
        except Exception as e:
            print(e)
            traceback.print_exc()
        else:
            pass
    return split_tracks


def dumpFiles(output_dir, mmsi_track):
    """Dump the msgs (of a given mmsi) to a file.

    Side effect ahead!"""
    mmsi, track = mmsi_track
    path = os.path.join(output_dir, "%s.txt" % mmsi)
    np.save(path, track.to_numpy())
    return mmsi_track

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build a set of trajectories from the raw ais signals")
    parser.add_argument("glob", type=str,
                        help="input files to parse in a glob format")

    parser.add_argument("ais_type", type=str,
                        choices=[GLOBAL, BRITTANY],
                        default="global",
                        help="ais type")

    # This will require to parially applied the filtered functions
    parser.add_argument('--output_dir', '-o', type=str,
                        help="Output dir",
                        default=cli_utils.lookup_env("OUTPUT_DIR", OUTPUT_DIR))

    parser.add_argument('--max_speed', '-M', type=int,
                        help="Maximum speed allowed (message is dropped if sog > max_speed)",
                        default=cli_utils.lookup_env_int("SPEED_MAX", SPEED_MAX))

    parser.add_argument('--interval_max', '-G', type=int,
                       help="Maximal duration between two consecutive points (a new trajectory is created otherwise)",
                       default=cli_utils.lookup_env_int("INTERVAL_MAX", config.interval_max))

    parser.add_argument("--min_timespan", "-t",
                        help="Max time span for a trajectory",
                        default=cli_utils.lookup_env_int("MIN_TIMESPAN", MIN_TIMESPAN))

    args = parser.parse_args()
    print(args)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    spark = SparkSession\
        .builder\
        .appName("count")\
        .getOrCreate()
    sc = spark.sparkContext

    rdd_ais = sc.textFile(args.glob)

    a = (
            rdd_ais.map(partial(decode, args.ais_type))
                .filter(lambda x: x)
                .map(extractKeys)
                .map(lambda msg: (msg.get("mmsi"), msg))
                .groupByKey()
                .flatMap(partial(buildTracks, (LAT_MIN, LAT_MAX), (LON_MIN, LON_MAX), MIN_TIMESPAN, args.interval_max , args.max_speed))
                .map(partial(dumpFiles, args.output_dir))
                .count()
        )
    print(a)

    spark.stop()