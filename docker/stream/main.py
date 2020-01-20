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


APP_NAME = "multitaskais_stream"

TRAJECTORIES = {}
LAST_CHECK = time.time()

## Duong Code <DC>

### std
import pickle

### 3rd party
from scipy import stats
import tensorflow as tf

### local
import contrario_utils
from flags_config_SESAME import config, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, SPEED_MAX
from models import vrnn
import runners
import utils

LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))
LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
LAT_RESO = config.anomaly_lat_reso
LON_RESO = config.anomaly_lon_reso
LAT_BIN = int(LAT_RANGE/LAT_RESO)
LON_BIN = int(LON_RANGE/LON_RESO)


DEFAULT_PARALLELISM = 12
num_parallel_calls=DEFAULT_PARALLELISM
## </DC>

## <DC>
def process_AIS_track(m_V):
    """
    Preprocess the AIS track.
    See the comments in each steps for details.

    ARGUMENTS:
        - m_V: a matrix, each row is an AIS message, in the following format:
               [LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI]
               HEADING, ROT and NAV_STT are not used at the moment.

        - Return:
        ## TODO: output's format.
    """

    ## Remove erroneous timestamps and erroneous speeds
    print(" Remove erroneous timestamps and erroneous speeds...")
    # Boundary
    lat_idx = np.logical_or((m_V[:,LAT] > LAT_MAX),
                            (m_V[:,LAT] < LAT_MIN))

    m_V = m_V[np.logical_not(lat_idx)]
    lon_idx = np.logical_or((m_V[:,LON] > LON_MAX),
                            (m_V[:,LON] < LON_MIN))
    m_V = m_V[np.logical_not(lon_idx)]
    # Abnormal timestamps
#    abnormal_timestamp_idx = np.logical_or((m_V[:,TIMESTAMP] > t_max),
#                                           (m_V[:,TIMESTAMP] < t_min))
#    m_V = m_V[np.logical_not(abnormal_timestamp_idx)]
    # Abnormal speeds
    abnormal_speed_idx = m_V[:,SOG] > SPEED_MAX
    m_V = m_V[np.logical_not(abnormal_speed_idx)]

    ## TODO: raise an error if len(m_V) == 0
    print("m_V's shape: ",m_V.shape)


    ## Cutting discontiguous voyages into contiguous ones
    print("Cutting discontiguous voyages into contiguous ones...")


    # Intervals between successive messages in a track
    intervals = m_V[1:,TIMESTAMP] - m_V[:-1,TIMESTAMP]
    idx = np.where(intervals > config.interval_max)[0]

    if len(idx) == 0:
        pass
    else:
        # NOTE(msimonin: WARNING: v is undefined)
        m_V = np.split(m_V,idx+1)[0] # Use the first contiguous segment only
        ## TODO: use all the contiguous segments

    print("m_V's shape: ",m_V.shape)

    ## Removing AIS track whose length is smaller than 20 or those last less than 4h
    print("Removing AIS track whose length is smaller than 20 or those last less than 4h...")

    duration = m_V[-1,TIMESTAMP] - m_V[0,TIMESTAMP]
    if (len(m_V) < 20) or (duration < 4*3600):
        print("Error 1!!!!!")
        ## TODO: raise an error

    ## Removing 'moored' or 'at anchor' voyages
    print("Removing 'moored' or 'at anchor' voyages...")

    d_L = float(len(m_V))

    if np.count_nonzero(m_V[:,NAV_STT] == 1)/d_L > 0.7   or np.count_nonzero(m_V[:,NAV_STT] == 5)/d_L > 0.7:
        print("Error 2!!!!!")
        ## TODO: raise an error
    sog_max = np.max(m_V[:,SOG])
    if sog_max < 1.0:
        print("Error 3!!!!!")
        ## TODO: raise an error
    if config.print_log:
        print("m_V's shape: ",m_V.shape)

    ## Sampling, resolution = 5 min
    print('Sampling...')

    sampling_track = np.empty((0, 9))
    for t in range(int(m_V[0,TIMESTAMP]), int(m_V[-1,TIMESTAMP]), 300): # 5 min
        tmp = utils.interpolate(t,m_V)
        if tmp is not None:
            sampling_track = np.vstack([sampling_track, tmp])
        else:
            sampling_track = None
            break
    if sampling_track is not None:
        m_V = sampling_track
    print("m_V's shape: ",m_V.shape)

    ## Removing 'low speed' tracks
    if config.print_log:
        print("Removing 'low speed' tracks...")
    d_L = float(len(m_V))
    if np.count_nonzero(m_V[:,SOG] < 2)/d_L > 0.8:
        print("Error 4!!!!!")
        ## TODO: raise an error
    print("m_V's shape: ",m_V.shape)

    print('Re-Splitting...')

    ## Split AIS track into small tracks whose duration <= 1 day
    idx = np.arange(0, len(m_V), 12*24)[1:]
    m_V = np.split(m_V,idx)[0]
    print("m_V's shape: ",m_V.shape)

    ## Normalisation
    print('Normalisation...')
    m_V[:,LAT] = (m_V[:,LAT] - LAT_MIN)/(LAT_MAX-LAT_MIN)
    m_V[:,LON] = (m_V[:,LON] - LON_MIN)/(LON_MAX-LON_MIN)
    m_V[:,SOG][m_V[:,SOG] > SPEED_MAX] = SPEED_MAX
    m_V[:,SOG] = m_V[:,SOG]/SPEED_MAX
    m_V[:,COG] = m_V[:,COG]/360.0
    print("m_V's shape: ",m_V.shape)

    return m_V


def build_and_load_GeoTrackNet(l_V):
    """
    Create the computational graph and load the pretrained weights of GeoTrackNet.
    ARGUMENTS:
        - l_V: a list of AIS track, each element is an AIS track (a matrix).
    """
    ## Create dataset and model.
    inputs, targets, lengths, mmsis, mean = create_AIS_dataset(l_V)

    generative_bias_init = -tf.log(1. / tf.clip_by_value(mean, 0.0001, 0.9999) - 1)
    generative_bias_init = tf.cast(generative_bias_init,tf.float32)

    generative_distribution_class = vrnn.ConditionalBernoulliDistribution
    model = vrnn.create_vrnn(inputs.get_shape().as_list()[2],
                             config.latent_size,
                             generative_distribution_class,
                             generative_bias_init=generative_bias_init,
                             raw_sigma_bias=0.5)

    ## Create the graph
    track_sample, track_true, log_weights, ll_per_t, ll_acc,_,_,_ =\
                             runners.create_eval_graph(inputs, targets, lengths, model, config)

    ## Reload the model
    saver = tf.train.Saver()
    sess = tf.train.SingularMonitoredSession()
    runners.wait_for_checkpoint(saver, sess, config.logdir)

    return sess, inputs, targets, lengths, mmsis, mean, log_weights, ll_per_t


def create_AIS_dataset(l_V):
    """
    Input pipeline for MultitaskAIS.
    ## TODO: l_V's format? output's format.
    """
    def sparse_AIS_to_dense(msgs_,num_timesteps, mmsis):
        """
        A function to create a suitable input format for MultitaskAIS.
        Do not modify this function
        """
    #        lat_bins = 200; lon_bins = 300; sog_bins = 30; cog_bins = 72
        def create_dense_vect(msg,lat_bins = 200, lon_bins = 300, sog_bins = 30 ,cog_bins = 72):
            lat, lon, sog, cog = msg[0], msg[1], msg[2], msg[3]
            data_dim = lat_bins + lon_bins + sog_bins + cog_bins
            dense_vect = np.zeros(data_dim)
            dense_vect[int(lat*lat_bins)] = 1.0
            dense_vect[int(lon*lon_bins) + lat_bins] = 1.0
            dense_vect[int(sog*sog_bins) + lat_bins + lon_bins] = 1.0
            dense_vect[int(cog*cog_bins) + lat_bins + lon_bins + sog_bins] = 1.0
            return dense_vect
    #    msgs_[msgs_ == 1] = 0.99999
        dense_msgs = []
        for msg in msgs_:
            dense_msgs.append(create_dense_vect(msg,
                                                lat_bins = config.lat_bins,
                                                lon_bins = config.lon_bins,
                                                sog_bins = config.sog_bins,
                                                cog_bins = config.cog_bins))
        dense_msgs = np.array(dense_msgs)
        return dense_msgs, num_timesteps, mmsis

    with open("./data/ct_2017010203_10_20/mean.pkl","rb") as f:
        mean = pickle.load(f)
    def aistrack_generator():
        for m_V in l_V:
            tmp = m_V[::2,[LAT,LON,SOG,COG]] # 10 min
            tmp[tmp == 1] = 0.99999
            yield tmp, len(tmp), m_V[0,MMSI]

    dataset = tf.data.Dataset.from_generator(
                                  aistrack_generator,
                                  output_types=(tf.float64, tf.int64, tf.int64))
    dataset = dataset.map(
            lambda msg_, num_timesteps, mmsis: tuple(tf.py_func(sparse_AIS_to_dense,
                                                   [msg_, num_timesteps, mmsis],
                                                   [tf.float64, tf.int64, tf.int64])),
                                                num_parallel_calls=num_parallel_calls)

    # Batch sequences togther, padding them to a common length in time.
    dataset = dataset.padded_batch(config.batch_size,
                                 padded_shapes=([None, 602], [], []))

    def process_AIS_batch(data, lengths, mmsis):
        """Create mean-centered and time-major next-step prediction Tensors."""
        data = tf.to_float(tf.transpose(data, perm=[1, 0, 2]))
        lengths = tf.to_int32(lengths)
        mmsis = tf.to_int32(mmsis)
        targets = data

        # Mean center the inputs.
        inputs = data - tf.constant(mean, dtype=tf.float32,
                                    shape=[1, 1, mean.shape[0]])
        # Shift the inputs one step forward in time. Also remove the last
        # timestep so that targets and inputs are the same length.
        inputs = tf.pad(data, [[1, 0], [0, 0], [0, 0]], mode="CONSTANT")[:-1]
        # Mask out unused timesteps.
        inputs *= tf.expand_dims(tf.transpose(
            tf.sequence_mask(lengths, dtype=inputs.dtype)), 2)
        return inputs, targets, lengths, mmsis

    dataset = dataset.map(process_AIS_batch,
                          num_parallel_calls=num_parallel_calls)


    dataset = dataset.prefetch(50)
    itr = dataset.make_one_shot_iterator()
    inputs, targets, lengths, mmsis = itr.get_next()
    return inputs, targets, lengths, mmsis, mean


def alert(trajectory):
    """Triggers an alert if trajectory is deemed abnormal

    Args:
        trajectory np.array: list of points (rows). Each point of the
        trajectory is composed of 9 columns
    """
    print(f"trajectory.shape: {trajectory.shape}")
    preprocessed_trajectory = process_AIS_track(copy.deepcopy(trajectory))
    l_V = [preprocessed_trajectory]

    ## Reset the computational graph.
    tf.reset_default_graph()
    tf.Graph().as_default()
    global_step = tf.train.get_or_create_global_step()

    ## Build and load the model
    sess, inputs, targets, lengths, mmsis, mean, log_weights, ll_per_t = build_and_load_GeoTrackNet(l_V)


    ## Loading the parameters of the distribution in each cell (calculated by the
    # tracks in the validation set)
    step = sess.run(global_step)
    save_dir = "results/"\
                + config.trainingset_path.split("/")[-2] + "/"\
                + "log_density-"\
                + os.path.basename(config.trainingset_name) + "-"\
                + os.path.basename(config.testset_name) + "-"\
                + str(config.latent_size) + "-"\
                + "missing_data-" + str(config.missing_data)\
                + "-step-"+str(step)\
                +"/"

    # NOTE(msimonin): DEAD CODE
    # m_map_ll_mean = np.load(save_dir+"map_ll_mean-"+str(LAT_RESO)+"-"+str(LON_RESO) + ".npy")
    # m_map_ll_std = np.load(save_dir+"map_ll_std-"+str(LAT_RESO)+"-"+str(LON_RESO) + ".npy")
    with open(save_dir+"map_ll-"+str(LAT_RESO)+"-"+str(LON_RESO)+".pkl","rb") as f:
        Map_ll = pickle.load(f)

    ## Calculate the log_prob of the track
    inp, tar, seq_len, mmsi, log_weights_np, ll_t =\
                sess.run([inputs, targets, lengths, mmsis, log_weights, ll_per_t])

    l_abnormal_track = []
    for idx_inbatch in range(tar.shape[1]):
        seq_len_d = seq_len[idx_inbatch]
        m_inp_sparse = np.nonzero(tar[:seq_len_d,idx_inbatch,:])[1].reshape(-1,4)
        m_log_weights_np =  log_weights_np[:seq_len_d,:,idx_inbatch]
        v_A = np.zeros(seq_len_d)

        for d_timestep in range(2*6,len(m_inp_sparse)):
            d_row = int(m_inp_sparse[d_timestep,0]*0.01/LAT_RESO)
            d_col = int((m_inp_sparse[d_timestep,1]-config.lat_bins)*0.01/LON_RESO)
            d_ll_t = np.mean(m_log_weights_np[d_timestep,:])

            ## KDE
            # Use KDE to estimate the distribution of log[p(x_t|h_t)] in each cell.
            l_local_log_prod = Map_ll[str(d_row)+","+str(d_col)]
            if len(l_local_log_prod) < 2:
                # Ignore cells that do not have enough data.
                v_A[d_timestep] = 2
            else:
                kernel = stats.gaussian_kde(l_local_log_prod)
                cdf = kernel.integrate_box_1d(-np.inf,d_ll_t)
                if cdf < 0.1:
                    v_A[d_timestep] = 1
        # log[p(x_t|h_t)] of the first timesteps of the tracks may not robust,
        # because h_t was initialized as a zeros.
        v_A = v_A[12:]
        v_anomalies = np.zeros(len(v_A))
        for d_i_4h in range(0,len(v_A)+1-24):
            v_A_4h = v_A[d_i_4h:d_i_4h+24]
            v_anomalies_i = contrario_utils.contrario_detection(v_A_4h,config.contrario_eps)
            v_anomalies[d_i_4h:d_i_4h+24][v_anomalies_i==1] = 1

        if len(contrario_utils.nonzero_segments(v_anomalies)) > 0:
            l_abnormal_track.append(mmsi[idx_inbatch])
        else:
            pass

    print("Number of abnormal tracks: ", len(l_abnormal_track))
    print(l_abnormal_track)
    if len(l_abnormal_track) > 0:
        return (trajectory, preprocessed_trajectory)
    return False
## </DC>


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
                    abnormal = await loop.run_in_executor(processes_pool, alert, l)
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