# Notes Regarding Running a Python Source File using Vertex AI Training

_Jorge III Altamirano-Astorga_

## Required Modules

### Additional

tqdm matplotlib dill

### Included in Pre-built Container

numpy tensorflow scikit_learn numpy pandas 

_Source: <https://cloud.google.com/vertex-ai/docs/training/pre-built-containers>._

## Execution

Run the python file in this directory

### Arguments:

-b 10 
-w 8 --sampling_rate=2 --stride=1
--steps=10
--epochs=100 
https://github.com/philwebsurfer/dlfinal/raw/main/data/data_5min.pickle.gz
gs://investigacion-sensor/output/

### Python Source Distribution

python3 setup.py sdist --formats=gztar

## GCP Shell

To check the execution of the job run:
gcloud ai custom-jobs list

## References

* <https://codelabs.developers.google.com/vertex_custom_training_prediction>

* <https://cloud.google.com/vertex-ai/docs/tutorials/image-recognition-custom/training#2_run_a_custom_training_pipeline>


