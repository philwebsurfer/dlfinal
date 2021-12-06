# Notes Regarding Running a Python Source File using Vertex AI Training

_Jorge III Altamirano-Astorga_

## Required Modules

### Additional

tqdm matplotlib dill

### Included in Pre-built Container

numpy tensorflow scikit_learn numpy pandas 

_Source: <https://cloud.google.com/vertex-ai/docs/training/pre-built-containers>._

## Building the Source Distribution

```
python3 setup.py sdist --formats=gztar
```

## Execution

Run the python file in this directory

### Arguments:

```
-b 128
-w 8 
--sampling_rate=2 
--stride=1 
--steps=10 
--epochs=100 
-m /gcs/investigacion-sensor/output/model_best01a.h5
https://github.com/philwebsurfer/dlfinal/raw/main/data/data_5min.pickle.gz
/gcs/investigacion-sensor/output/
```

### Python Module

This is the module fed into Python:

```
trainer.task
```

This trainslates into running as:
```
python3 -m trainer.task -b 10 -w 8 --sampling_rate=2 --stride=1 --steps=10 --epochs=100 -m ../model_best01a.h5  https://github.com/philwebsurfer/dlfinal/raw/main/data/data_5min.pickle.gz gs://investigacion-sensor/output/model
```


## GCP Shell

To check the execution of the job run:

```
gcloud config set ai/region us-central1
gcloud ai custom-jobs list
```

## Hyperparameter Tuning Job

```Model``` parameter for the hyperparameter tuning. With a categorical settings and it should go through all the models:

```
/gcs/investigacion-sensor/output/ParNet00.h5,/gcs/investigacion-sensor/output/model_best01a.h5,/gcs/investigacion-sensor/output/model_best03a.h5,/gcs/investigacion-sensor/output/model_best03b.h5,/gcs/investigacion-sensor/output/model_conv00.h5,/gcs/investigacion-sensor/output/model_conv02.h5,/gcs/investigacion-sensor/output/model_dnn00.h5,/gcs/investigacion-sensor/output/model_dnn01.h5,/gcs/investigacion-sensor/output/model_dnn02.h5,/gcs/investigacion-sensor/output/model_lstm00.h5,/gcs/investigacion-sensor/output/model_lstm02.h5,/gcs/investigacion-sensor/output/model_rnn00.h5,/gcs/investigacion-sensor/output/model_rnn02.h5
```

And the minimizing metric is ```mse```.

## References

* <https://codelabs.developers.google.com/vertex_custom_training_prediction>

* <https://cloud.google.com/vertex-ai/docs/training/create-training-pipeline#custom-job-model-upload>

* <https://cloud.google.com/vertex-ai/docs/tutorials/image-recognition-custom/training>

* <https://cloud.google.com/vertex-ai/docs/training/pre-built-containers>

* <https://cloud.google.com/vertex-ai/docs/training/create-python-pre-built-container>

* <https://cloud.google.com/vertex-ai/docs/tutorials/image-recognition-custom/training>

* <https://cloud.google.com/vertex-ai/docs/training/code-requirements>

* <https://docs.python.org/3/distutils/sourcedist.html>

* <https://docs.python.org/3/distutils/setupscript.html>

* <https://cloud.google.com/vertex-ai/pricing#americas>

* <https://cloud.google.com/compute/gpus-pricing>
