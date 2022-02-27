#!/usr/bin/env python3
#@author: Jorge III Altamirano-Astorga, Luz Aurora Hernández-Martínez
import tensorflow as tf
#from tensorflow.keras.models import Sequential, Model, load_model
#from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, \
#  SimpleRNN, Input, Conv1D, Flatten
#from tensorflow.keras.regularizers import l1, l2
#from tensorflow.keras.utils import plot_model
import re, os, sys, shelve, time, dill, io, logging
import argparse #args from cli
from pickle import PicklingError
from dill import Pickler, Unpickler
shelve.Pickler = Pickler
shelve.Unpickler = Unpickler
import pandas as pd
import numpy as np
#from tqdm.keras import TqdmCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse

tf.get_logger().setLevel('ERROR')

def train_model(x_train, y_train,
                x_test, y_test,
                kernel="rbf", degree=3, 
                coef0=.0, regularization=1.,
                njobs=1, metrics=['mse'], 
                verbose=0, output_datastore=""):
  #model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  #logging.info(f'Saving plot file: {model.name}.png')
  #plot_model(model, to_file=os.path.join(output_datastore, 
  #    f"{model.name}.png"), dpi=72, rankdir="TB", show_shapes=True, 
  #    expand_nested=True)
  #cbk = TqdmCallback()
  tiempo = time.time()
  logging.info('Training start...')
  trained_model = SVR(kernel=kernel, degree=degree, 
    coef0=coef0, C=regularization).fit(x_train, y_train)
  #history = model.fit(train_data, validation_data=validation_data,
  #                    epochs=epochs, steps_per_epoch=steps_per_epoch, 
  #                    batch_size=batch_size, verbose=verbose, callbacks=[cbk])
  tiempo = time.time() - tiempo
  logging.info(f"Processing Time: {tiempo:.2f} seconds.")

  #### Start Section: Save the Model
  base_dir = os.path.join(output_datastore, 
    f"svr-k{kernel}-0{coef0}-C{regularization}")
  logging.info(f'Saving model file: {base_dir}.model.dill')
  trained_model.dump(f"{base_dir}.model.dill")
  logging.info(f'Saving time file: {base_dir}.time.dill')
  dill.dump(tiempo, open(f"{base_dir}.time.dill", 'wb'))

  logging.info("Calculating mse and mae...")
  mse_ = mse(trained_model.predict(x_train), y_train)
  mae_ = mae(trained_model.predict(x_train), y_train)
  val_mse = mse(trained_model.predict(x_test), y_test)
  val_mae = mse(trained_model.predict(x_test), y_test)
  history = {
    "history": {
      "loss": [mse_],
      "mse": [mse_],
      "mae": [mae_],
      "val_loss": [val_mse],
      "val_mse": [val_mse],
      "val_mae": [val_mae]
    }
  }
  logging.info("Done calculating mse and mae!")
  logging.info(f'Saving history file: {base_dir}.hist.dill')
  dill.dump(history.history, open(f"{base_dir}.hist.dill", 'wb'))
  #### End Section: Save the Model
  #### Start Section: Hyperparameter Tuning
  try: 
      import hypertune

      hpt = hypertune.HyperTune()
      hpt.report_hyperparameter_tuning_metric(
              hyperparameter_metric_tag='mse',
              metric_value=history.history["val_loss"][-1],
              global_step=epochs
              )
  except Exception as e:
      logging.error("Start error of hypertune")
      logging.error(e)
      logging.error("End error of hypertune")
      pass
  #### End Section: Hyperparameter Tuning
  return history

def execute_train(window_size_days=2, stride=1, sampling_rate=1, 
        batch_size=128, kernel="rbf", degree=3, coef0=.0, regularization=1.,
        njobs=1, input_dataset="", output_datastore=""):
  # Data Prep
  logging.info(f'Loading input dataset: {input_dataset}')
  data =  pd.read_pickle(input_dataset)
  data = data[~data.isna().any(axis=1)]
  excluded_columns = ["iaqAccuracy", "wind_speed", "wind_deg"]
  train, test = train_test_split(data[[x 
                                          for x in data.columns 
                                          if x not in excluded_columns]], 
                                 train_size=0.7, random_state=175904, shuffle=False)
  # Scaling
  scaler = MinMaxScaler()
  scaler_f = scaler.fit(train)
  train2 = scaler_f.transform(train)
  test2 = scaler_f.transform(test)
  X_cols = [i for i, x in enumerate(train.columns) 
            if x not in ["IAQ", "gasResistance"]]
  Y_cols = [i for i, x in enumerate(train.columns) 
            if x in ["IAQ", "gasResistance"]]
  X_train = train2[:, X_cols]
  Y_train = train2[:, Y_cols]
  X_test  = test2[:, X_cols]
  Y_test = test2[:, Y_cols]

  timedelta_minutes = (data.index[-1] - data.index[-2]).seconds//60
  past = int(window_size_days) * 24 * 60 // timedelta_minutes
  past = 240
  stride = 30
  sampling_rate = 1
  batch_size = 1
  train3_iaq = tf.keras.preprocessing.timeseries_dataset_from_array(
    X_train, 
    Y_train[:, 1],
    sequence_length=past,
    sampling_rate=sampling_rate,
    sequence_stride=stride,
    batch_size=batch_size,
    seed=175904
  )
  test3_iaq = tf.keras.preprocessing.timeseries_dataset_from_array(
    X_test, 
    Y_test[:, 1],
    sequence_length=past,
    sampling_rate=sampling_rate,
    sequence_stride=stride,
    batch_size=batch_size,
    seed=175904
  )

  #logging.info(f"Trying to load {model_file}")
  #model = load_model(model_file)
  logging.info(f'Converting time series tensor into a multivariate ts format...')
  logging.info("... for training data...")
  x_train = []
  y_train = []
  for batch in train3_iaq:
    x_batch, y_batch = batch
    x_train.append(x_batch)
    y_train.append(y_batch)
  
  # Convert into numpy array
  logging.debug(f"x_train.shape = {np.array(x_train).shape}")
  logging.debug(f"y_train.shape = {np.array(y_train).shape}")
  logging.debug(f"train2.shape = {train2.shape}")
  x_train = np.array(x_train).reshape(x_train.shape[0],
        -1, past, train2.shape[1])
  y_train = np.array(y_train).reshape(-1, 1)
  logging.info("... Done training data!")
  logging.info("... for test data...")
  x_test = []
  y_test = []
  for batch in test3_iaq:
    x_batch, y_batch = batch
    x_test.append(x_batch)
    y_test.append(y_batch)
  logging.debug(f"x_test.shape =  {np.array(x_test).shape}")
  logging.debug(f"y_test.shape =  {np.array(x_test).shape}")
  logging.debug(f"test2.shape =  {test2.shape}")
  # Convert into numpy array
  x_test = np.array(x_test).reshape(-1, past, test2.shape[1])
  y_test = np.array(y_test).reshape(-1, 1)
  logging.info("... Done validation data!")
  logging.info(f'Done converting time series tensor into a multivariate ts format!')

  logging.info(f'Training SVR...')
  trained_model = train_model(x_train, y_train,
                              x_test, y_test,
                              metrics=["mse", "mae"],
                              kernel=kernel, 
                              degree=degree, 
                              coef0=coef0, 
                              regularization=regularization, 
                              njobs=njobs, 
                              output_datastore=output_datastore)
  logging.info("Done training SVR!")
  #trained_model = train_model(model, train3_iaq,
  #                            validation_data=test3_iaq,
  #                            metrics=["mse", "mae"],
  #                            epochs=epochs, steps_per_epoch=steps, 
  #                            batch_size=batch_size, 
  #                            output_datastore=output_datastore)
  #### Start Section: Save the Model
  base_dir = os.path.join(output_datastore, model.name)
  timeseries_params = {
          "window_size_days": window_size_days, 
          "stride": stride, 
          "sampling_rate": sampling_rate,
          "batch_size": batch_size,
          "model_n_params": f"{model.count_params():3,}"
          }
  dill.dump(timeseries_params, open(f"{base_dir}.tsparams.dill", 'wb'))
  scaler_path = os.path.join(output_datastore, "scaler.dill")
  scaler_iaq = MinMaxScaler().fit(train[["IAQ"]])
  dill.dump(scaler_iaq, open(scaler_path, 'wb'))
  #### End Section: Save the Model
    
def usage(argv):
    parser = argparse.ArgumentParser(
        description="""
        Creates a trained model based on data from a URL pickle and trains 
        it using Machine Learning Techniques.""",
        epilog="Example: %(prog)s https://data.example.com/data/data.pickle.gz . ", 
        prefix_chars='-')
            
    parser.add_argument('--batch_size', '-b', nargs='?', default=128, type=int,
                       help="""Batch size. Default 128.""")
    #parser.add_argument('--epochs', '-e', nargs='?', default=10, type=int,
    #        help="""Training epochs. Default: 10.""")
    #parser.add_argument('--steps', '-t', nargs='?', default=100, type=int,
    #        help="""Training steps per epoch. Default: 100.""")
    parser.add_argument('--window_size_days', '-w', nargs='?', default=2, type=int,
            help="""Window Size in Days. Default: 2.""")
    parser.add_argument('--stride', '-s', nargs='?', default=2, type=int,
            help="""Window Size in Days. Default: 2.""")
    parser.add_argument('--sampling_rate', '-r', nargs='?', default=1, type=int,
            help="""Window Size in Days. Default:2.""")
    parser.add_argument('--debug', '-d', action="store_false", default=False,
                       help='Debugging and Verbose Messages.')
    parser.add_argument('--kernel', '-k', nargs='?', default="rbf",
            help="""Specifies the kernel type to be used in the algorithm. 
Can be ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’. Default: rbf.""")
    parser.add_argument('--degree', '-f', nargs='?', default=3, type=int,
            help="""Degree of the polynomial kernel function. Default: 3.""")
    parser.add_argument('--coef0', '-0', nargs='?', default=0.0, type=float,
            help="""Independent term in kernel function. Default: 0.0""")
    parser.add_argument('--regularization', '-C', nargs='?', default=1., type=float,
            help="""Regularization parameter. Must be strictly positive. Default: 1.0""")
    parser.add_argument('--njobs', '-n', nargs='?', default=1., type=int,
            help="""Parallelization level, ideally match number of cpu. Default: 1""")
    parser.add_argument('input_dataset', nargs=1,
                        help="""Input dataset URL or Location of the File.""")
    parser.add_argument('output_datastore', nargs=1, default="", 
                        help="""Output dataset URL or Location of the File.
                        """)
    return parser.parse_args()

def main(argv):
    ### read cli arguments
    args = usage(argv)
    #print(args)

    ## Google Environment
    #if 'AIP_MODEL_DIR' not in os.environ:
    #    raise KeyError(
    #        'The `AIP_MODEL_DIR` environment variable has not been' +
    #        'set. See https://cloud.google.com/ai-platform-unified/docs/tutorials/image-recognition-custom/training'
    #    )
    output_dir = args.output_datastore[0].strip()
    #model_file = args.model[0].strip()

    logging.basicConfig()
    if args.debug or True:
        logging.getLogger().setLevel(logging.DEBUG)
    logging.StreamHandler(sys.stdout)
    
    execute_train(window_size_days=args.window_size_days, 
            batch_size=int(args.batch_size),
            stride=args.stride,
            sampling_rate=args.sampling_rate,
            kernel=args.kernel,
            degree=int(args.degree),
            coef0=float(args.coef0),
            regularization=float(args.regularization),
            njobs=int(args.njobs),
            input_dataset=args.input_dataset[0],
            output_datastore=output_dir,
           )
    
    exit(0)
    
    
if __name__ == "__main__":
    main(sys.argv)
