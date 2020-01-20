import argparse
import asyncio
from collections import defaultdict
import copy
import concurrent.futures
import os
import time
import traceback

import faust
import numpy as np
from sesamelib.sesame_faust import MutatedDynamicMessage
from sesamelib.multitask import Track, SpeedError, AreaError, MaxIntervalError
import sesamelib.utils as cli_utils

# Import some bits of Duong code
from online_detection import alert
from flags_config_SESAME import config, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, SPEED_MAX


APP_NAME = "multitaskais_stream"

TRAJECTORIES = {}
LAST_CHECK = time.time()
MIN_TIMESPAN = 4 * 60 * 60


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Alert if a ship enters one of the indexed EEZs")
    parser.add_argument("--bootstrap_servers", "-b",
                        help="Kafka bootstrap servers",
                        action="append",
                        default=cli_utils.lookup_env_array("BOOTSTRAP_SERVERS", "localhost:9092"))
    parser.add_argument("--topic", "-t",
                        help="Kafka topic to get ais messages from",
                        default=cli_utils.lookup_env("TOPIC", "ais.dynamic"))
    parser.add_argument("--min_timespan", "-m",
                        help="Max time span for a trajectory",
                        default=cli_utils.lookup_env_int("MIN_TIMESPAN", MIN_TIMESPAN))
    args = parser.parse_args()

    loop = asyncio.get_event_loop()

    app = faust.App(
        APP_NAME,
        loop=loop,
        broker="kafka://{}".format(",".join(args.bootstrap_servers))
    )

    topic = app.topic(args.topic, key_type=str, value_type=MutatedDynamicMessage)
    out_topic = app.topic(APP_NAME)

    channel = app.channel(value_type=str)

    processes_pool = concurrent.futures.ProcessPoolExecutor(10)

    @app.timer(interval=5.0)
    async def healthcheck():
        global LAST_CHECK
        now = time.time()
        interval = now - LAST_CHECK
        LAST_CHECK = now
        drift =  interval - 5
        print(f"HB:  drift of {drift} / #trajectories {len(list(TRAJECTORIES.keys()))}")


    def new_traj():
        return Track(
                    min_timespan=2*args.min_timespan,
                    lat_min_max=(LAT_MIN,LAT_MAX),
                    lon_min_max=(LON_MIN, LON_MAX),
                    max_speed=SPEED_MAX,
                    max_interval=config.interval_max
                )


    @app.agent(topic)
    async def buffer(msgs):
        async for msg in msgs:
            global TRAJECTORIES
            TRAJECTORIES.setdefault(msg.mmsi, new_traj())
            try:
                TRAJECTORIES[msg.mmsi].add(msg)
                if TRAJECTORIES[msg.mmsi].exceed():
                    l = TRAJECTORIES[msg.mmsi].to_numpy()
                    abnormal = await loop.run_in_executor(processes_pool, alert, [l])
                    del TRAJECTORIES[msg.mmsi]
            except MaxIntervalError:
                print(f"Creating a new track for {msg.mmsi}")
                del TRAJECTORIES[msg.mmsi]
            except Exception as e:
                print(e)
                traceback.print_exc()
            else:
                pass


    app.finalize()
    worker = faust.Worker(app,
                          loglevel="INFO")

    worker.execute_from_commandline()

    # some cleaning
    processes_pool.shutdown()