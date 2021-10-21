#!/usr/bin/env python3
#@author: Jorge III Altamirano-Astorga
import re, os, sys, shelve, time, dill, io, logging
import argparse #args from cli
from pickle import PicklingError
from dill import Pickler, Unpickler
shelve.Pickler = Pickler
shelve.Unpickler = Unpickler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, \
  SimpleRNN, Input, Conv1D, Flatten
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
tf.get_logger().setLevel('ERROR')

def performance_plot(history, a=None, b=None, 
                    metrics=["accuracy", "val_accuracy"],
                    plot_validation=True,
                    title="Performance Plots for Model."):
  """
  Prints performance plot from a, to b on a history dict.
  
  Inputs:
  history: dict containing "loss" and "accuracy" keys
  a: epoch start
  b. last epoch
  metrics: plot these metrics (train and validation). Always 2.
  plot_validation: boolean indicating if validation data should be plotted.
  a: from this epoch
  b: to this epoch    
  """
  if a is None:
      a = 0
  if b is None:
      b = len(history['loss'])
  a = np.min((a,b))
  b = np.max((a,b))

  imgrows = (len(metrics) + 1) / 2
  imgrows = np.round(imgrows, 0)
  imgrows = int(imgrows)
  #print(imgrows)

  # Plot loss
  plt.figure(figsize=(14, 5
                      *imgrows))
  plt.suptitle(title)
  plt.subplot(imgrows, 2, 1)
  plt.title('Loss')
  plt.plot(history['loss'][a:b], label='Training', linewidth=2)
  if plot_validation:
    plt.plot(history['val_loss'][a:b], label='Validation', linewidth=2)
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  quantiles = np.quantile(range(a, b), 
                          [.2, .4, .6, .8]).round(0).astype(int)
  quantiles = np.insert(quantiles, 0, [a])
  quantiles += 1
  quantiles = np.append(quantiles, [b-1])
  plt.xticks(ticks=quantiles-a,
              labels=quantiles)
  plt.grid(True)

  # Plot accuracy
  for i, metric in enumerate(metrics): 
    #print(f"metric: {metric}, i: {i}")
    #print(f"mean metric: {np.mean(history[metric])}")
    plt.subplot(imgrows, 2, i+2)
    plt.title(metric)
    plt.plot(history[metric][a:b], label='Training', 
              linewidth=2)
    if plot_validation:
      plt.plot(history["val_" + metric][a:b], 
                label='Validation', linewidth=2)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    #plt.xlim(a, b)
    #print(range(0, b-a))
    plt.xticks(ticks=quantiles-a, 
                labels=quantiles)
    plt.grid(True)

  plt.show()

def train_model(model, train_data,  validation_data,
                epochs=10, batch_size=128, 
                steps_per_epoch=100, loss='mse', optimizer='adam', 
                metrics=['mse'], verbose=0, output_datastore=""):
  model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  plot_model(model, to_file=os.path.join(output_datastore, 
      f"{model.name}.png"), dpi=72, rankdir="TB", show_shapes=True, 
      expand_nested=True)
  cbk = TqdmCallback()
  tiempo = time.time()
  #print(f"Tipo: {type(batch_size)} // batch_size={batch_size}")
  #print(f"Tipo: {type(steps_per_epoch)} // steps={steps_per_epoch}")
  #print(f"Tipo: {type(epochs)} // epochs={epochs}")
  history = model.fit(train_data, validation_data=validation_data,
                      epochs=epochs, steps_per_epoch=steps_per_epoch, 
                      batch_size=batch_size, verbose=verbose, callbacks=[cbk])
  tiempo = time.time() - tiempo
  logging.info(f"Processing Time: {tiempo:.2f} segundos.")

  #### Start Section: Save the Model
  base_dir = os.path.join(output_datastore, model.name)
  model.save(f"{base_dir}.h5")
  dill.dump(tiempo, open(f"{base_dir}.time.dill", 'wb'))
  dill.dump(history.history, open(f"{base_dir}.hist.dill", 'wb'))
  #### End Section: Save the Model
  return history

def execute_train(window_size_days=2, stride=1, sampling_rate=1, 
        batch_size=128, steps=100, epochs=10, model_file="",
        input_dataset="", output_datastore=""):
  # Data Prep
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

  model_best01a = load_model(model_file)
  ## Model Creation
  ## model_best01a = Sequential(name="model_best01a")
  ## model_best01a.add(Input(shape=(X_train.shape[0], X_train.shape[1], ), 
  ##                        name="input00"))
  ## model_best01a.add(Conv1D(512, X_train.shape[1], activation='relu', name="conv00"))
  ## model_best01a.add(Dropout(0.3, name="dropout00"))
  ## model_best01a.add(Dense(units=512, activation='relu', name="dnn"))
  ## model_best01a.add(Dropout(0.3))
  ## model_best01a.add(Dense(units=256, activation='relu'))
  ## model_best01a.add(Dropout(0.5))
  ## model_best01a.add(Dense(units=256, activation='relu'))
  ## model_best01a.add(Dense(units=1, activation=None, name="output"))

  trained_model01a = train_model(model_best01a, train3_iaq,
                              validation_data=test3_iaq,
                              metrics=["mse", "mae"],
                              epochs=epochs, steps_per_epoch=steps, 
                              batch_size=batch_size, 
                              output_datastore=output_datastore)
  #### Start Section: Save the Model
  base_dir = os.path.join(output_datastore, model_best01a.name)
  timeseries_params = {
          "window_size_days": window_size_days, 
          "stride": stride, 
          "sampling_rate": sampling_rate,
          "batch_size": batch_size
          }
  dill.dump(timeseries_params, open(f"{base_dir}.tsparams.dill", 'wb'))
  #### End Section: Save the Model
    
def usage(argv):
    parser = argparse.ArgumentParser(
        description="""
        Creates a model based on data from a URL pickle and trains 
        it using Neural Networks.""",
        epilog="Example: %(prog)s https://data.example.com/data/data.pickle.gz . ", 
        prefix_chars='-')
            
    parser.add_argument('--batch_size', '-b', nargs='?', default=128, type=int,
                       help="""Batch size. Default 128.""")
    parser.add_argument('--epochs', '-e', nargs='?', default=10, type=int,
            help="""Training epochs. Default: 10.""")
    parser.add_argument('--steps', '-t', nargs='?', default=100, type=int,
            help="""Training steps per epoch. Default: 100.""")
    parser.add_argument('--window_size_days', '-w', nargs='?', default=2, type=int,
            help="""Window Size in Days. Default: 2.""")
    parser.add_argument('--stride', '-s', nargs='?', default=2, type=int,
            help="""Window Size in Days. Default: 2.""")
    parser.add_argument('--sampling_rate', '-r', nargs='?', default=1, type=int,
            help="""Window Size in Days. Default:2.""")
    parser.add_argument('--debug', '-d', action="store_false", default=False,
                       help='Debugging and Verbose Messages.')
    parser.add_argument('--model', '-m', nargs=1, required=True,
                        help="""H5 model file.""")
    parser.add_argument('input_dataset', nargs=1,
                        help="""Input dataset URL or Location of the File.""")
    #parser.add_argument('output_datastore', nargs=1, default="", 
    #                    help="""Output dataset URL or Location of the File.
    #                    """)
    return parser.parse_args()

def main(argv):
    ### read cli arguments
    args = usage(argv)
    #print(args)

    ## Google Environment
    if 'AIP_MODEL_DIR' not in os.environ:
        raise KeyError(
            'The `AIP_MODEL_DIR` environment variable has not been' +
            'set. See https://cloud.google.com/ai-platform-unified/docs/tutorials/image-recognition-custom/training'
        )
    output_directory = os.environ['AIP_MODEL_DIR']
    model_file = os.path.join(output_directory, args.model[0].strip())
    logging.info(f"training python script: AIP_MODEL_DIR={output_directory}")
    logging.info(f"training python script: model_file={model_file}")
    
    #execute_train(window_size_days=2, stride=1, sampling_rate=1):
    execute_train(window_size_days=args.window_size_days, 
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            steps=int(args.steps),
            stride=args.stride,
            sampling_rate=args.sampling_rate,
            input_dataset=args.input_dataset[0],
            output_datastore=output_directory,
            model_file=model_file
           )
    
    exit(0)
    
    
if __name__ == "__main__":
    main(sys.argv)
